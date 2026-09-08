# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Correctness and API-contract tests for gfx950 SonicMoE A16W16/A16W4."""

import json
import math
import threading
import weakref
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.moe.moe_2stage_a16wmix.gemm1 import compile_gemm1_a16w4_port
from kernels.moe.sonic import (
    SonicMoE,
    SonicMoEConfig,
    SonicMoEForwardState,
    SonicMoEWeights,
    SonicMoEWorkspace,
    _get_stage1_launcher,
    _get_stage1_training_launcher,
    _get_stage2_launcher,
    _quantize_mxfp4_weight,
    _stage2_stages,
    _training_stage1_tuning,
    _validate_training_preactivation_extent,
    prepare_sonic_bf16_weights,
    prepare_sonic_fp16_weights,
    prepare_sonic_mxfp4_weights,
    sonic_moe_mxfp4_reference,
    sonic_moe_reference,
)
from kernels.moe.sonic_autotune import SonicMoEAutotuner, default_sonic_moe_candidates

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

TOKENS = 7
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 128
NUM_EXPERTS = 4
TOP_K = 2


def _config(**overrides):
    values = {
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "tile_m": 16,
        "tile_n": 128,
        "tile_k": 128,
    }
    values.update(overrides)
    return SonicMoEConfig(**values)


def _gfx950_device():
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU is required")
    arch = str(get_rocm_arch())
    if "gfx950" not in arch:
        pytest.skip(f"SonicMoE BF16 test requires gfx950, found {arch}")
    return torch.device("cuda")


def _make_case(tokens=TOKENS, seed=17, activation="swiglu", dtype=torch.bfloat16):
    device = _gfx950_device()
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((tokens, HIDDEN_SIZE), device=device, dtype=torch.float32, generator=generator).to(dtype)
    stage1_size = INTERMEDIATE_SIZE * (2 if activation in ("swiglu", "geglu", "reglu") else 1)
    w1 = (
        torch.randn(
            (NUM_EXPERTS, stage1_size, HIDDEN_SIZE),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        / math.sqrt(HIDDEN_SIZE)
    ).to(dtype)
    w2 = (
        torch.randn(
            (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        / math.sqrt(INTERMEDIATE_SIZE)
    ).to(dtype)
    router_logits = torch.randn((tokens, NUM_EXPERTS), device=device, dtype=torch.float32, generator=generator).to(
        dtype
    )
    return x, w1, w2, router_logits


def _topk_from_logits(router_logits, config):
    scores = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(scores, config.top_k, dim=-1)
    if config.renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_ids.to(torch.int32), topk_weights.contiguous()


def _assert_close(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-2, atol=5e-2)


def _prepare_dense_weights(w1, w2, config, *, b1=None, b2=None):
    prepare = prepare_sonic_fp16_weights if config.compute_dtype == "fp16" else prepare_sonic_bf16_weights
    return prepare(w1, w2, config, b1=b1, b2=b2)


def _fixed_topk_preactivation_oracle(x, w1, topk_ids, *, b1=None, interleaved=False):
    """Independent logical-weight GEMM oracle for compact route-order state."""

    tokens, top_k = topk_ids.shape
    selected_w1 = w1[topk_ids.long()].float().reshape(tokens * top_k, w1.shape[1], w1.shape[2])
    route_x = x[:, None, :].expand(tokens, top_k, x.shape[1]).reshape(tokens * top_k, x.shape[1]).float()
    preactivation = torch.bmm(selected_w1, route_x.unsqueeze(-1)).squeeze(-1)
    if b1 is not None:
        preactivation = preactivation + b1[topk_ids.long()].float().reshape(tokens * top_k, w1.shape[1])
    preactivation = preactivation.to(torch.bfloat16).view(tokens, top_k, -1)
    if interleaved:
        gate, up = preactivation.chunk(2, dim=-1)
        preactivation = torch.stack((gate, up), dim=-1).reshape(tokens, top_k, -1)
    return preactivation


def test_sonic_moe_bf16_forward_matches_reference():
    config = _config()
    x, w1, w2, router_logits = _make_case()
    prepared = prepare_sonic_bf16_weights(w1, w2, config)
    op = SonicMoE(config, prepared)

    expected = sonic_moe_reference(x, w1, w2, router_logits, config)
    actual = op(x, router_logits)
    torch.cuda.synchronize()
    _assert_close(actual, expected)

    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    actual_topk = op.forward_topk(x, topk_ids, topk_weights)
    torch.cuda.synchronize()
    _assert_close(actual_topk, expected)

    out = torch.empty(expected.shape, device=expected.device, dtype=torch.bfloat16)
    returned = op(x, router_logits, out=out)
    torch.cuda.synchronize()
    assert returned is out
    _assert_close(out, expected)


@pytest.mark.parametrize("interleaved_w1", (False, True), ids=("separate", "interleaved"))
@pytest.mark.parametrize("has_bias", (False, True), ids=("no-bias", "bias"))
def test_sonic_moe_training_forward_state_matches_route_order_gemm(interleaved_w1, has_bias):
    config = _config(stage1_lds_swizzle=True)
    x, w1, w2, router_logits = _make_case(seed=271)
    generator = torch.Generator(device=x.device).manual_seed(273)
    b1 = (
        torch.randn(
            (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE),
            dtype=torch.bfloat16,
            device=x.device,
            generator=generator,
        )
        / 8
        if has_bias
        else None
    )
    b2 = (
        torch.randn(
            (NUM_EXPERTS, HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=x.device,
            generator=generator,
        )
        / 8
        if has_bias
        else None
    )
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config, b1=b1, b2=b2))
    out = torch.empty_like(x)

    returned, state = op.forward_topk_training(
        x,
        topk_ids,
        topk_weights,
        out=out,
        interleaved_w1=interleaved_w1,
    )
    expected_state = _fixed_topk_preactivation_oracle(
        x,
        w1,
        topk_ids,
        b1=b1,
        interleaved=interleaved_w1,
    )
    expected_out = sonic_moe_reference(x, w1, w2, router_logits, config, b1=b1, b2=b2)
    torch.cuda.synchronize()

    assert returned is out
    assert isinstance(state, SonicMoEForwardState)
    assert state.preactivation.shape == (TOKENS, TOP_K, 2 * INTERMEDIATE_SIZE)
    assert state.preactivation.dtype == torch.bfloat16
    assert state.preactivation.is_contiguous()
    assert state.tokens == TOKENS
    assert state.hidden_size == HIDDEN_SIZE
    assert state.intermediate_size == INTERMEDIATE_SIZE
    assert state.num_experts == NUM_EXPERTS
    assert state.top_k == TOP_K
    assert state.activation == "swiglu"
    assert state.compute_dtype == "bf16"
    assert state.interleaved_w1 is interleaved_w1
    assert state.has_bias is has_bias
    assert state.producer_stream == int(torch.cuda.current_stream(x.device).cuda_stream)
    assert state.ready_event.query()
    assert op.workspace is not None
    assert state.preactivation.untyped_storage().data_ptr() not in op.workspace.storage_ptrs
    with pytest.raises(FrozenInstanceError):
        state.tokens = 1
    torch.testing.assert_close(state.preactivation.float(), expected_state.float(), rtol=3e-2, atol=5e-2)
    _assert_close(out, expected_out)


def test_sonic_moe_training_states_are_invocation_owned_and_not_overwritten():
    config = _config()
    x_a, w1, w2, logits_a = _make_case(seed=277)
    x_b, _, _, logits_b = _make_case(seed=281)
    ids_a, weights_a = _topk_from_logits(logits_a, config)
    ids_b, weights_b = _topk_from_logits(logits_b, config)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))

    _, state_a = op.forward_topk_training(x_a, ids_a, weights_a)
    state_a_snapshot = state_a.preactivation.clone()
    _, state_b = op.forward_topk_training(x_b, ids_b, weights_b)
    torch.cuda.synchronize()

    assert state_a.preactivation.data_ptr() != state_b.preactivation.data_ptr()
    assert state_a.ready_event is not state_b.ready_event
    assert torch.equal(state_a.preactivation, state_a_snapshot)
    expected_b = _fixed_topk_preactivation_oracle(x_b, w1, ids_b)
    torch.testing.assert_close(state_b.preactivation.float(), expected_b.float(), rtol=3e-2, atol=5e-2)
    assert op.workspace is not None
    assert state_a.preactivation.untyped_storage().data_ptr() not in op.workspace.storage_ptrs
    assert state_b.preactivation.untyped_storage().data_ptr() not in op.workspace.storage_ptrs


@pytest.mark.parametrize("tokens", (1, 129), ids=("direct-t1", "multiphase-sort"))
def test_sonic_moe_training_state_covers_fixed_topk_sort_paths(tokens):
    config = _config(stage1_k_wave=2 if tokens == 1 else 1)
    x, w1, w2, router_logits = _make_case(tokens=tokens, seed=279 + tokens)
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))

    _, state = op.forward_topk_training(x, topk_ids, topk_weights)
    expected = _fixed_topk_preactivation_oracle(x, w1, topk_ids)
    torch.cuda.synchronize()

    assert state.preactivation.shape == (tokens, TOP_K, 2 * INTERMEDIATE_SIZE)
    torch.testing.assert_close(state.preactivation.float(), expected.float(), rtol=3e-2, atol=5e-2)


@pytest.mark.parametrize("training", (False, True), ids=("inference", "training"))
@pytest.mark.parametrize("tokens", (1, 129), ids=("direct-t1", "multiphase"))
def test_sonic_moe_fixed_topk_frequency_matches_routes(training, tokens):
    """Both public fixed-K entry points fill caller-owned frequency storage."""

    config = _config(stage1_k_wave=2 if tokens == 1 else 1)
    x, w1, w2, router_logits = _make_case(tokens=tokens, seed=307 + tokens)
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    expected = sonic_moe_reference(x, w1, w2, router_logits, config)
    expected_frequency = torch.bincount(
        topk_ids.flatten().long(),
        minlength=config.num_experts,
    ).to(torch.int32)
    frequency = torch.full_like(expected_frequency, -1)
    out = torch.empty_like(x)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))

    if training:
        returned, state = op.forward_topk_training(
            x,
            topk_ids,
            topk_weights,
            out=out,
            expert_frequency_out=frequency,
        )
        assert state.preactivation.shape[0] == tokens
    else:
        returned = op.forward_topk(
            x,
            topk_ids,
            topk_weights,
            out=out,
            expert_frequency_out=frequency,
        )
    torch.cuda.synchronize()

    assert returned is out
    assert torch.equal(frequency, expected_frequency)
    _assert_close(out, expected)


@pytest.mark.parametrize("training", (False, True), ids=("inference", "training"))
def test_sonic_moe_fixed_topk_frequency_rejects_aliases(training):
    config = _config()
    x, w1, w2, router_logits = _make_case(seed=439)
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    out = torch.empty_like(x)

    def invoke(frequency):
        if training:
            return op.forward_topk_training(
                x,
                topk_ids,
                topk_weights,
                out=out,
                expert_frequency_out=frequency,
            )
        return op.forward_topk(
            x,
            topk_ids,
            topk_weights,
            out=out,
            expert_frequency_out=frequency,
        )

    input_alias = topk_ids.flatten()[: config.num_experts]
    with pytest.raises(ValueError, match="must not alias an input or output"):
        invoke(input_alias)

    output_alias = out.view(torch.int32).flatten()[: config.num_experts]
    with pytest.raises(ValueError, match="must not alias an input or output"):
        invoke(output_alias)

    workspace = op.reserve(x.shape[0])
    with pytest.raises(ValueError, match="must not alias internal workspace storage"):
        invoke(workspace.expert_frequency)


@pytest.mark.parametrize("training", (False, True), ids=("inference", "training"))
def test_sonic_moe_fixed_topk_frequency_cross_stream_lifetime(training):
    config = _config()
    x, w1, w2, router_logits = _make_case(seed=443)
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    expected_frequency = torch.bincount(
        topk_ids.flatten().long(),
        minlength=config.num_experts,
    ).to(torch.int32)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    producer = torch.cuda.Stream(device=x.device)
    consumer = torch.cuda.Stream(device=x.device)
    producer.wait_stream(torch.cuda.current_stream(x.device))

    with torch.cuda.stream(producer):
        frequency = torch.empty(config.num_experts, dtype=torch.int32, device=x.device)
        if training:
            output, state = op.forward_topk_training(
                x,
                topk_ids,
                topk_weights,
                expert_frequency_out=frequency,
            )
        else:
            output = op.forward_topk(
                x,
                topk_ids,
                topk_weights,
                expert_frequency_out=frequency,
            )
            state = None

    consumer.wait_stream(producer)
    with torch.cuda.stream(consumer):
        observed = frequency.clone()
        checksum = output.float().sum()
        if state is not None:
            checksum = checksum + state.preactivation.float().sum()
    torch.cuda.current_stream(x.device).wait_stream(consumer)

    assert torch.equal(observed, expected_frequency)
    assert torch.isfinite(checksum)


def test_sonic_moe_training_state_records_current_stream_and_ready_event():
    config = _config()
    x, w1, w2, router_logits = _make_case(seed=283)
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    producer = torch.cuda.Stream(device=x.device)
    producer.wait_stream(torch.cuda.current_stream(x.device))

    with torch.cuda.stream(producer):
        out, state = op.forward_topk_training(x, topk_ids, topk_weights)
    assert state.producer_stream == int(producer.cuda_stream)

    consumer = torch.cuda.Stream(device=x.device)
    consumer.wait_event(state.ready_event)
    with torch.cuda.stream(consumer):
        consumed = state.preactivation.float().sum() + out.float().sum()
    torch.cuda.current_stream(x.device).wait_stream(consumer)
    assert torch.isfinite(consumed)


def test_sonic_moe_training_forward_rejects_capture_without_state_slot(monkeypatch):
    config = _config()
    x, w1, w2, router_logits = _make_case(seed=293)
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="graph-private preallocated state slot"):
        op.forward_topk_training(x, topk_ids, topk_weights)


def test_sonic_moe_training_state_extent_stays_below_masked_store_sentinel():
    max_tokens = 0x7FFFFFFF // 4
    _validate_training_preactivation_extent(max_tokens, 1, 1)

    with pytest.raises(ValueError, match="signed 32-bit masked-store byte-offset limit"):
        _validate_training_preactivation_extent(max_tokens + 1, 1, 1)


def test_sonic_moe_route_state_compile_rejects_situv2():
    with pytest.raises(AssertionError, match="does not support act='situv2'"):
        compile_gemm1_a16w4_port(
            BM=16,
            D_HIDDEN=128,
            D_INTER=128,
            NE=1,
            TOPK=1,
            TILE_N=128,
            TILE_K=128,
            act="situv2",
            w_dtype="bf16",
            a_dtype="bf16",
            logical_dense_weight=True,
            round_preact_bf16=True,
            store_route_preactivation=True,
        )


def test_sonic_moe_inference_stage1_launcher_does_not_enable_dual_store(monkeypatch):
    import kernels.moe.sonic as sonic_module

    config = _config()
    compile_kwargs = []

    def fake_compile(**kwargs):
        compile_kwargs.append(kwargs)
        return object()

    monkeypatch.setattr(sonic_module, "compile_gemm1_a16w4_port", fake_compile)
    _get_stage1_launcher.cache_clear()
    _get_stage1_training_launcher.cache_clear()
    try:
        inference = _get_stage1_launcher(config, 2, "bf16", False, 0)
        assert inference is _get_stage1_launcher(config, 2, "bf16", False, 0)
        assert compile_kwargs[-1].get("store_route_preactivation", False) is False
        assert compile_kwargs[-1]["skip_epilogue_id_reload"] is False
        assert compile_kwargs[-1]["a_lds_swizzle"] is False
        assert _get_stage1_launcher.cache_info().currsize == 1
        assert _get_stage1_training_launcher.cache_info().currsize == 0

        training = _get_stage1_training_launcher(config, 2, False, True, 0)
        assert training is _get_stage1_training_launcher(config, 2, False, True, 0)
        assert compile_kwargs[-1]["store_route_preactivation"] is True
        assert compile_kwargs[-1]["route_preactivation_interleaved"] is True
        assert compile_kwargs[-1]["a_lds_swizzle"] is False
        assert _get_stage1_launcher.cache_info().currsize == 1
        assert _get_stage1_training_launcher.cache_info().currsize == 1
    finally:
        _get_stage1_launcher.cache_clear()
        _get_stage1_training_launcher.cache_clear()


def test_sonic_moe_stage1_padding_store_is_an_inference_cache_key(monkeypatch):
    import kernels.moe.sonic as sonic_module

    compile_kwargs = []

    def fake_compile(**kwargs):
        compile_kwargs.append(kwargs)
        return object()

    monkeypatch.setattr(sonic_module, "compile_gemm1_a16w4_port", fake_compile)
    _get_stage1_launcher.cache_clear()
    try:
        masked = _get_stage1_launcher(_config(), 0, "bf16", False, 0)
        padded = _get_stage1_launcher(
            _config(stage1_write_padded_rows=True), 0, "bf16", False, 0
        )

        assert masked is not padded
        assert [call["skip_epilogue_id_reload"] for call in compile_kwargs] == [
            False,
            True,
        ]
        assert _get_stage1_launcher.cache_info().currsize == 2
    finally:
        _get_stage1_launcher.cache_clear()


def test_sonic_moe_stage1_lds_swizzle_is_a_launcher_cache_key(monkeypatch):
    import kernels.moe.sonic as sonic_module

    compile_kwargs = []

    def fake_compile(**kwargs):
        compile_kwargs.append(kwargs)
        return object()

    monkeypatch.setattr(sonic_module, "compile_gemm1_a16w4_port", fake_compile)
    _get_stage1_launcher.cache_clear()
    _get_stage1_training_launcher.cache_clear()
    try:
        linear = _get_stage1_launcher(_config(), 0, "bf16", False, 0)
        swizzled = _get_stage1_launcher(
            _config(stage1_lds_swizzle=True), 0, "bf16", False, 0
        )
        training = _get_stage1_training_launcher(
            _config(stage1_lds_swizzle=True), 0, False, False, 0
        )

        assert linear is not swizzled
        assert [call["a_lds_swizzle"] for call in compile_kwargs] == [
            False,
            True,
            True,
        ]
        assert _get_stage1_launcher.cache_info().currsize == 2
        assert _get_stage1_training_launcher.cache_info().currsize == 1
        assert training is not swizzled
    finally:
        _get_stage1_launcher.cache_clear()
        _get_stage1_training_launcher.cache_clear()


def test_sonic_moe_training_stage1_t4096_policy_is_targeted():
    throughput = SonicMoEConfig(
        hidden_size=4096,
        intermediate_size=2048,
        num_experts=64,
        top_k=8,
        tile_m=128,
        tile_n=256,
        tile_k=64,
        down_tile_m=128,
        down_tile_n=128,
        down_tile_k=64,
        stage1_k_wave=1,
        stage2_xcd_swizzle=8,
        renormalize=False,
    )

    assert _training_stage1_tuning(throughput, 4096, False) == (128, 2)
    assert _training_stage1_tuning(throughput, 2048, False) == (256, None)
    assert _training_stage1_tuning(throughput, 4096, True) == (256, None)
    assert _training_stage1_tuning(replace(throughput, tile_n=128), 4096, False) == (
        128,
        None,
    )
    assert _training_stage1_tuning(replace(throughput, waves_per_eu=1), 4096, False) == (
        256,
        1,
    )
    assert _training_stage1_tuning(replace(throughput, stage1_b_cache_mod=2), 4096, False) == (
        256,
        None,
    )
    assert _training_stage1_tuning(replace(throughput, stage1_xcd_swizzle=1), 4096, False) == (
        256,
        None,
    )
    assert _training_stage1_tuning(replace(throughput, down_tile_m=256), 4096, False) == (
        256,
        None,
    )


def test_sonic_moe_training_stage1_launcher_accepts_private_overrides(monkeypatch):
    import kernels.moe.sonic as sonic_module

    config = _config()
    compile_calls = []

    def fake_compile(**kwargs):
        compile_calls.append(kwargs)
        return object()

    monkeypatch.setattr(sonic_module, "compile_gemm1_a16w4_port", fake_compile)
    _get_stage1_training_launcher.cache_clear()
    try:
        launcher = _get_stage1_training_launcher(
            config,
            2,
            False,
            False,
            0,
            64,
            2,
        )
        assert launcher is _get_stage1_training_launcher(
            config,
            2,
            False,
            False,
            0,
            64,
            2,
        )
        assert compile_calls[-1]["TILE_N"] == 64
        assert compile_calls[-1]["waves_per_eu"] == 2
    finally:
        _get_stage1_training_launcher.cache_clear()


def test_sonic_moe_training_stage1_private_override_is_numerically_exact(monkeypatch):
    import kernels.moe.sonic as sonic_module

    config = _config()
    x, w1, w2, router_logits = _make_case(seed=313)
    ids, weights = _topk_from_logits(router_logits, config)
    prepared = prepare_sonic_bf16_weights(w1, w2, config)
    op = SonicMoE(config, prepared)
    expected_out = sonic_moe_reference(x, w1, w2, router_logits, config)
    expected_state = _fixed_topk_preactivation_oracle(x, w1, ids)

    monkeypatch.setattr(
        sonic_module,
        "_training_stage1_tuning",
        lambda _config, _tokens, _has_bias: (64, 2),
    )
    output, state = op.forward_topk_training(x, ids, weights)

    _assert_close(output, expected_out)
    torch.testing.assert_close(
        state.preactivation.float(),
        expected_state.float(),
        rtol=3e-2,
        atol=5e-2,
    )


def test_sonic_moe_training_forward_rejects_unsupported_contracts():
    fp16_config = _config(compute_dtype="fp16")
    x, w1, w2, router_logits = _make_case(seed=307, dtype=torch.float16)
    ids, weights = _topk_from_logits(router_logits, fp16_config)
    fp16_op = SonicMoE(fp16_config, prepare_sonic_fp16_weights(w1, w2, fp16_config))
    with pytest.raises(NotImplementedError, match="dense BF16"):
        fp16_op.forward_topk_training(x, ids, weights)

    relu_config = _config(activation="relu")
    x, w1, w2, router_logits = _make_case(seed=311, activation="relu")
    ids, weights = _topk_from_logits(router_logits, relu_config)
    relu_op = SonicMoE(relu_config, prepare_sonic_bf16_weights(w1, w2, relu_config))
    with pytest.raises(NotImplementedError, match="activation='swiglu'"):
        relu_op.forward_topk_training(x, ids, weights)


def test_sonic_moe_fp16_forward_matches_reference():
    config = _config(compute_dtype="fp16", stage1_lds_swizzle=True)
    x, w1, w2, router_logits = _make_case(dtype=torch.float16)
    prepared = prepare_sonic_fp16_weights(w1, w2, config)
    op = SonicMoE(config, prepared)

    expected = sonic_moe_reference(x, w1, w2, router_logits, config)
    actual = op(x, router_logits)
    torch.cuda.synchronize()
    _assert_close(actual, expected)

    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    actual_topk = op.forward_topk(x, topk_ids, topk_weights)
    torch.cuda.synchronize()
    _assert_close(actual_topk, expected)

    out = torch.empty_like(expected)
    returned = op(x, router_logits, out=out)
    torch.cuda.synchronize()
    assert returned is out
    _assert_close(out, expected)


def test_sonic_moe_stage1_k_wave2_matches_reference():
    config = _config(
        stage1_k_wave=2,
        stage1_write_padded_rows=True,
        stage1_lds_swizzle=True,
    )
    x, w1, w2, router_logits = _make_case(seed=147)
    expected = sonic_moe_reference(x, w1, w2, router_logits, config)
    actual = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))(x, router_logits)
    torch.cuda.synchronize()
    _assert_close(actual, expected)


def test_sonic_moe_stage1_padding_store_k_wave4_matches_reference():
    config = _config(
        tile_m=32,
        tile_n=64,
        tile_k=64,
        down_tile_m=32,
        down_tile_n=128,
        down_tile_k=128,
        stage1_k_wave=4,
        stage1_write_padded_rows=True,
        stage1_lds_swizzle=True,
    )
    x, w1, w2, router_logits = _make_case(seed=148)
    expected = sonic_moe_reference(x, w1, w2, router_logits, config)
    actual = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))(
        x, router_logits
    )
    torch.cuda.synchronize()
    _assert_close(actual, expected)


@pytest.mark.parametrize(
    ("compute_dtype", "torch_dtype"),
    (("bf16", torch.bfloat16), ("fp16", torch.float16)),
)
def test_sonic_moe_dense_stage1_tile_k64_matches_k128(compute_dtype, torch_dtype):
    """Pin the compact dense-fragment map used by gfx950 Stage1 K64."""

    config64 = _config(
        tile_m=32,
        tile_n=64,
        tile_k=64,
        down_tile_n=128,
        down_tile_k=128,
        compute_dtype=compute_dtype,
        stage1_lds_swizzle=True,
    )
    config128 = replace(config64, tile_k=128)
    x, w1, w2, router_logits = _make_case(seed=149, dtype=torch_dtype)
    topk_ids, topk_weights = _topk_from_logits(router_logits, config64)

    op64 = SonicMoE(config64, _prepare_dense_weights(w1, w2, config64))
    op128 = SonicMoE(config128, _prepare_dense_weights(w1, w2, config128))
    out64 = op64.forward_topk(x, topk_ids, topk_weights)
    out128 = op128.forward_topk(x, topk_ids, topk_weights)
    torch.cuda.synchronize()

    expected = sonic_moe_reference(x, w1, w2, router_logits, config64)
    _assert_close(out64, expected)
    _assert_close(out128, expected)

    def unsort_intermediate(op):
        workspace = op.workspace
        assert workspace is not None
        padded = int(workspace.num_valid_ids[0].item())
        packed = workspace.sorted_token_ids[:padded]
        token = packed & 0x00FFFFFF
        valid = token < TOKENS
        token = token[valid].to(torch.long)
        slot = ((packed[valid] >> 24) & 0xFF).to(torch.long)
        routes = torch.empty(
            (TOKENS, TOP_K, INTERMEDIATE_SIZE),
            dtype=torch_dtype,
            device=x.device,
        )
        routes[token, slot] = workspace.intermediate[:padded][valid]
        return routes

    _assert_close(unsort_intermediate(op64), unsort_intermediate(op128))


def test_sonic_moe_stage1_lds_swizzle_single_k_tile_matches_reference():
    config = _config(
        tile_k=256,
        down_tile_k=128,
        stage1_lds_swizzle=True,
    )
    x, w1, w2, router_logits = _make_case(seed=150)
    expected = sonic_moe_reference(x, w1, w2, router_logits, config)
    actual = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))(
        x, router_logits
    )
    torch.cuda.synchronize()
    _assert_close(actual, expected)


@pytest.mark.parametrize(
    "stage1_tile_m,stage2_tile_m",
    ((32, 128), (64, 128), (128, 64), (48, 64)),
)
def test_sonic_moe_independent_stage_tile_m_matches_fixed_and_ragged_reference(
    stage1_tile_m,
    stage2_tile_m,
):
    """A route tile may be subdivided by either grouped GEMM independently."""

    config = _config(tile_m=stage1_tile_m, down_tile_m=stage2_tile_m)
    x, w1, w2, router_logits = _make_case(seed=150 + stage1_tile_m + stage2_tile_m)
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    expected = sonic_moe_reference(x, w1, w2, router_logits, config)

    actual = op.forward_topk(x, topk_ids, topk_weights)
    torch.cuda.synchronize()
    _assert_close(actual, expected)
    assert op.workspace is not None
    assert op.workspace.route_tile_m == math.lcm(stage1_tile_m, stage2_tile_m)
    assert op.workspace.max_padded_tokens % config.route_tile_m == 0
    assert op.workspace.stage1_max_m_blocks == op.workspace.max_padded_tokens // stage1_tile_m
    assert op.workspace.stage2_max_m_blocks == op.workspace.max_padded_tokens // stage2_tile_m

    router_actual = op(x, router_logits)
    torch.cuda.synchronize()
    _assert_close(router_actual, expected)

    token_indices = torch.arange(TOKENS, dtype=torch.int32, device=x.device).repeat_interleave(TOP_K)
    ragged = op.forward_routes(
        x,
        token_indices,
        topk_ids.reshape(-1),
        topk_weights.reshape(-1),
    )
    torch.cuda.synchronize()
    _assert_close(ragged, expected)


@pytest.mark.parametrize(
    ("weight_dtype", "compute_dtype", "torch_dtype"),
    (
        ("bf16", "bf16", torch.bfloat16),
        ("fp16", "fp16", torch.float16),
        ("mxfp4", "bf16", torch.bfloat16),
    ),
)
def test_sonic_moe_reduce_mode_fixed_topk_matches_reference_and_route_sum(
    weight_dtype,
    compute_dtype,
    torch_dtype,
):
    config = _config(
        stage2_output_mode="reduce",
        compute_dtype=compute_dtype,
        stage1_write_padded_rows=True,
    )
    x, w1, w2, router_logits = _make_case(seed=151, dtype=torch_dtype)
    generator = torch.Generator(device=x.device).manual_seed(157)
    b1 = (
        torch.randn(
            (NUM_EXPERTS, config.stage1_projection_size),
            dtype=torch.float32,
            device=x.device,
            generator=generator,
        )
        / 8
    ).to(torch_dtype)
    b2 = (
        torch.randn(
            (NUM_EXPERTS, HIDDEN_SIZE),
            dtype=torch.float32,
            device=x.device,
            generator=generator,
        )
        / 8
    ).to(torch_dtype)
    if weight_dtype == "mxfp4":
        prepared = prepare_sonic_mxfp4_weights(w1, w2, config, b1=b1, b2=b2)
        expected = sonic_moe_mxfp4_reference(x, w1, w2, router_logits, config, b1=b1, b2=b2)
    else:
        prepared = _prepare_dense_weights(w1, w2, config, b1=b1, b2=b2)
        expected = sonic_moe_reference(x, w1, w2, router_logits, config, b1=b1, b2=b2)
    op = SonicMoE(config, prepared)
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)

    router_out = torch.empty_like(x)
    actual_router = op(x, router_logits, out=router_out)
    torch.cuda.synchronize()
    assert actual_router is router_out
    assert op.workspace is not None and op.workspace.route_output is not None
    assert op.workspace.route_output.shape == (TOKENS, TOP_K, HIDDEN_SIZE)
    assert torch.equal(router_out, op.workspace.route_output.float().sum(dim=1).to(torch_dtype))
    _assert_close(router_out, expected)

    topk_out = torch.empty_like(x)
    actual_topk = op.forward_topk(x, topk_ids, topk_weights, out=topk_out)
    torch.cuda.synchronize()
    assert actual_topk is topk_out
    assert op.workspace is not None and op.workspace.route_output is not None
    assert torch.equal(topk_out, op.workspace.route_output.float().sum(dim=1).to(torch_dtype))
    _assert_close(topk_out, expected)


def test_sonic_moe_reduce_config_keeps_ragged_routes_atomic():
    config = _config(stage2_output_mode="reduce")
    x, w1, w2, router_logits = _make_case(seed=163)
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    token_indices = torch.arange(TOKENS, dtype=torch.int32, device=x.device).repeat_interleave(config.top_k)

    expected = sonic_moe_reference(x, w1, w2, router_logits, config)
    actual = op.forward_routes(
        x,
        token_indices,
        topk_ids.reshape(-1).contiguous(),
        topk_weights.reshape(-1).contiguous(),
    )
    torch.cuda.synchronize()

    assert op.workspace is not None
    assert op.workspace.routes == TOKENS * TOP_K
    assert op.workspace.route_output is None
    _assert_close(actual, expected)


def test_sonic_moe_fp16_reference_quantizes_fp32_source_weights():
    config = _config(compute_dtype="fp16")
    source_x, source_w1, source_w2, router_logits = _make_case(seed=119, dtype=torch.float32)
    x = source_x.to(torch.float16)

    expected = sonic_moe_reference(x, source_w1, source_w2, router_logits, config)
    quantized_expected = sonic_moe_reference(
        x,
        source_w1.to(torch.float16),
        source_w2.to(torch.float16),
        router_logits,
        config,
    )
    actual = SonicMoE(
        config,
        prepare_sonic_fp16_weights(source_w1, source_w2, config),
    )(x, router_logits)
    torch.cuda.synchronize()

    assert torch.equal(expected, quantized_expected)
    _assert_close(actual, expected)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_sonic_moe_dense_i64_activation_bias_fixed_and_flat_routes(dtype):
    """Cover the legacy Triton H=128/I=64/E=4/K=2 correctness shape."""

    device = _gfx950_device()
    hidden_size, intermediate_size = 128, 64
    config = SonicMoEConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=4,
        top_k=2,
        tile_m=32,
        tile_n=64,
        tile_k=128,
        down_tile_n=128,
        down_tile_k=64,
        activation="geglu",
        compute_dtype="fp16" if dtype == torch.float16 else "bf16",
    )
    generator = torch.Generator(device=device).manual_seed(137)
    x = torch.randn((TOKENS, hidden_size), dtype=torch.float32, device=device, generator=generator).to(dtype)
    w1 = (
        torch.randn(
            (config.num_experts, config.stage1_projection_size, hidden_size),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        / math.sqrt(hidden_size)
    ).to(dtype)
    w2 = (
        torch.randn(
            (config.num_experts, hidden_size, intermediate_size),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        / math.sqrt(intermediate_size)
    ).to(dtype)
    b1 = (
        torch.randn(
            (config.num_experts, config.stage1_projection_size),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        / 8
    ).to(dtype)
    b2 = (
        torch.randn(
            (config.num_experts, hidden_size),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        / 8
    ).to(dtype)
    router_logits = torch.randn(
        (TOKENS, config.num_experts),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(dtype)
    prepared = _prepare_dense_weights(w1, w2, config, b1=b1, b2=b2)
    op = SonicMoE(config, prepared)
    expected = sonic_moe_reference(x, w1, w2, router_logits, config, b1=b1, b2=b2)
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)

    actual_fixed = op.forward_topk(x, topk_ids, topk_weights)
    token_indices = torch.arange(TOKENS, dtype=torch.int32, device=device).repeat_interleave(config.top_k)
    actual_flat = op.forward_routes(
        x,
        token_indices,
        topk_ids.reshape(-1).contiguous(),
        topk_weights.reshape(-1).contiguous(),
    )
    torch.cuda.synchronize()

    _assert_close(actual_fixed, expected)
    _assert_close(actual_flat, expected)


@pytest.mark.parametrize(
    "activation",
    ("swiglu", "geglu", "reglu", "gelu_tanh_approx", "relu", "silu", "relu_sq"),
)
def test_sonic_moe_fp16_activation_variants_with_bias_and_flat_routes(activation):
    config = _config(activation=activation, compute_dtype="fp16")
    x, w1, w2, router_logits = _make_case(seed=113, activation=activation, dtype=torch.float16)
    generator = torch.Generator(device=x.device).manual_seed(127)
    b1 = (
        torch.randn(
            (NUM_EXPERTS, config.stage1_projection_size),
            device=x.device,
            dtype=torch.float32,
            generator=generator,
        )
        / 8
    ).to(torch.float16)
    b2 = (
        torch.randn(
            (NUM_EXPERTS, HIDDEN_SIZE),
            device=x.device,
            dtype=torch.float32,
            generator=generator,
        )
        / 8
    ).to(torch.float16)
    op = SonicMoE(
        config,
        _prepare_dense_weights(w1, w2, config, b1=b1, b2=b2),
    )
    expected = sonic_moe_reference(x, w1, w2, router_logits, config, b1=b1, b2=b2)
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)

    actual_fixed = op.forward_topk(x, topk_ids, topk_weights)
    token_indices = torch.arange(TOKENS, dtype=torch.int32, device=x.device).repeat_interleave(config.top_k)
    actual_flat = op.forward_routes(
        x,
        token_indices,
        topk_ids.reshape(-1).contiguous(),
        topk_weights.reshape(-1).contiguous(),
    )
    torch.cuda.synchronize()

    _assert_close(actual_fixed, expected)
    _assert_close(actual_flat, expected)


@pytest.mark.parametrize(
    "activation",
    ("swiglu", "geglu", "reglu", "gelu_tanh_approx", "relu", "silu", "relu_sq"),
)
def test_sonic_moe_bf16_activation_variants_fixed_and_flat_routes(activation):
    config = _config(activation=activation)
    x, w1, w2, router_logits = _make_case(seed=83, activation=activation)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    expected = sonic_moe_reference(x, w1, w2, router_logits, config)
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)

    actual_fixed = op.forward_topk(x, topk_ids, topk_weights)
    token_indices = torch.arange(TOKENS, dtype=torch.int32, device=x.device).repeat_interleave(config.top_k)
    actual_flat = op.forward_routes(
        x,
        token_indices,
        topk_ids.reshape(-1).contiguous(),
        topk_weights.reshape(-1).contiguous(),
    )
    torch.cuda.synchronize()

    _assert_close(actual_fixed, expected)
    _assert_close(actual_flat, expected)


@pytest.mark.parametrize("activation", ("geglu", "relu_sq"))
def test_sonic_moe_mxfp4_activation_variants(activation):
    config = _config(activation=activation, stage1_lds_swizzle=True)
    x, w1, w2, router_logits = _make_case(seed=89, activation=activation)
    prepared = prepare_sonic_mxfp4_weights(w1, w2, config)
    expected = sonic_moe_mxfp4_reference(x, w1, w2, router_logits, config)
    actual = SonicMoE(config, prepared)(x, router_logits)
    torch.cuda.synchronize()
    _assert_close(actual, expected)


@pytest.mark.parametrize("stage1_write_padded_rows", (False, True))
@pytest.mark.parametrize("activation", ("swiglu", "relu"))
def test_sonic_moe_bf16_bias_matches_reference_fixed_and_flat_routes(
    activation,
    stage1_write_padded_rows,
):
    config = _config(
        activation=activation,
        stage1_write_padded_rows=stage1_write_padded_rows,
        stage1_lds_swizzle=True,
    )
    x, w1, w2, router_logits = _make_case(seed=97, activation=activation)
    generator = torch.Generator(device=x.device).manual_seed(101)
    b1 = (
        torch.randn(
            (NUM_EXPERTS, config.stage1_projection_size),
            device=x.device,
            dtype=torch.float32,
            generator=generator,
        )
        / 8
    ).to(torch.bfloat16)
    b2 = (
        torch.randn(
            (NUM_EXPERTS, HIDDEN_SIZE),
            device=x.device,
            dtype=torch.float32,
            generator=generator,
        )
        / 8
    ).to(torch.bfloat16)
    prepared = prepare_sonic_bf16_weights(w1, w2, config, b1=b1, b2=b2)
    assert prepared.has_bias
    expected = sonic_moe_reference(x, w1, w2, router_logits, config, b1=b1, b2=b2)
    op = SonicMoE(config, prepared)

    actual_fixed = op(x, router_logits)
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    token_indices = torch.arange(TOKENS, dtype=torch.int32, device=x.device).repeat_interleave(config.top_k)
    actual_flat = op.forward_routes(
        x,
        token_indices,
        topk_ids.reshape(-1).contiguous(),
        topk_weights.reshape(-1).contiguous(),
    )
    torch.cuda.synchronize()

    _assert_close(actual_fixed, expected)
    _assert_close(actual_flat, expected)


def test_sonic_moe_mxfp4_bias_matches_reference():
    config = _config(activation="geglu")
    x, w1, w2, router_logits = _make_case(seed=103, activation="geglu")
    b1 = torch.linspace(
        -0.25,
        0.25,
        NUM_EXPERTS * config.stage1_projection_size,
        device=x.device,
        dtype=torch.bfloat16,
    ).view(NUM_EXPERTS, config.stage1_projection_size)
    b2 = torch.linspace(
        0.125,
        -0.125,
        NUM_EXPERTS * HIDDEN_SIZE,
        device=x.device,
        dtype=torch.bfloat16,
    ).view(NUM_EXPERTS, HIDDEN_SIZE)
    prepared = prepare_sonic_mxfp4_weights(w1, w2, config, b1=b1, b2=b2)
    expected = sonic_moe_mxfp4_reference(x, w1, w2, router_logits, config, b1=b1, b2=b2)
    actual = SonicMoE(config, prepared)(x, router_logits)
    torch.cuda.synchronize()
    _assert_close(actual, expected)


def test_sonic_moe_bias_ragged_duplicate_zero_score_and_empty_routes():
    device = _gfx950_device()
    config = _config(activation="relu")
    x = torch.zeros((3, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    w1 = torch.zeros(
        (NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )
    w2 = torch.zeros(
        (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )
    b1 = torch.ones((NUM_EXPERTS, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device=device)
    b2 = torch.stack(
        [torch.full((HIDDEN_SIZE,), expert + 1, dtype=torch.bfloat16, device=device) for expert in range(NUM_EXPERTS)]
    )
    op = SonicMoE(
        config,
        prepare_sonic_bf16_weights(w1, w2, config, b1=b1, b2=b2),
    )
    token_indices = torch.tensor([0, 0, 2, 2], dtype=torch.int32, device=device)
    expert_indices = torch.tensor([1, 1, 0, 2], dtype=torch.int32, device=device)
    route_weights = torch.tensor([0.25, -0.5, 0.0, 1.5], device=device)

    actual = op.forward_routes(x, token_indices, expert_indices, route_weights)
    torch.cuda.synchronize()
    expected = torch.zeros_like(x)
    expected[0].fill_(-0.5)
    expected[2].fill_(4.5)
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual[1]) == 0

    empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
    empty = op.forward_routes(x, empty_i32, empty_i32, empty_f32)
    torch.cuda.synchronize()
    assert torch.count_nonzero(empty) == 0


def test_sonic_moe_activation_preserves_legacy_bf16_preactivation_rounding():
    device = _gfx950_device()
    config = _config(num_experts=1, top_k=1, activation="relu_sq")
    value = 1.5078125
    x = torch.zeros((1, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    x[0, 0] = value
    w1 = torch.zeros((1, INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
    w1[0, :, 0] = value
    w2 = torch.zeros((1, HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=torch.bfloat16, device=device)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))

    op.forward_topk(
        x,
        torch.zeros((1, 1), dtype=torch.int32, device=device),
        torch.ones((1, 1), dtype=torch.float32, device=device),
    )
    torch.cuda.synchronize()

    assert op.workspace is not None
    preactivation = (torch.tensor(value) * torch.tensor(value)).to(torch.bfloat16).float()
    expected = preactivation.square().to(torch.bfloat16)
    without_preactivation_rounding = torch.tensor(value * value).square().to(torch.bfloat16)
    assert expected.item() != without_preactivation_rounding.item()
    assert torch.equal(
        op.workspace.intermediate[0],
        torch.full((INTERMEDIATE_SIZE,), expected.item(), dtype=torch.bfloat16, device=device),
    )


def test_sonic_moe_ragged_routes_match_reference_and_frequency():
    """Flat routes allow missing tokens, duplicate edges, and arbitrary weights."""

    config = _config(
        stage1_write_padded_rows=True,
        stage1_lds_swizzle=True,
    )
    x, w1, w2, _ = _make_case(seed=19)
    prepared = prepare_sonic_bf16_weights(w1, w2, config)
    op = SonicMoE(config, prepared)
    token_indices = torch.tensor(
        [0, 0, 1, 3, 3, 3, 5, 6, 6, 6, 6],
        dtype=torch.int32,
        device=x.device,
    )
    expert_indices = torch.tensor(
        [1, 3, 0, 2, 2, 1, 3, 0, 1, 2, 3],
        dtype=torch.int32,
        device=x.device,
    )
    route_weights = torch.tensor(
        [0.7, 0.3, 1.2, -0.4, 0.6, 0.0, 0.9, 0.1, 0.2, 0.3, 0.4],
        dtype=torch.float32,
        device=x.device,
    )
    frequency = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=x.device)

    expected = torch.zeros_like(x, dtype=torch.float32)
    x_f32, w1_f32, w2_f32 = x.float(), w1.float(), w2.float()
    for route in range(route_weights.numel()):
        token = int(token_indices[route])
        expert = int(expert_indices[route])
        gate_up = (w1_f32[expert] @ x_f32[token]).to(torch.bfloat16).float()
        gate, up = gate_up.split(INTERMEDIATE_SIZE)
        activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).float()
        projected = w2_f32[expert] @ activated
        expected[token].add_(projected * route_weights[route])
    expected = expected.to(torch.bfloat16)

    actual = op.forward_routes(
        x,
        token_indices,
        expert_indices,
        route_weights,
        expert_frequency_out=frequency,
    )
    torch.cuda.synchronize()

    expected_frequency = torch.bincount(expert_indices.to(torch.int64), minlength=NUM_EXPERTS).to(torch.int32)
    assert torch.equal(frequency, expected_frequency)
    assert op.workspace is not None
    assert op.workspace.routes == route_weights.numel()
    expected_blocks = sum(
        (int(count) + config.route_tile_m - 1) // config.route_tile_m
        for count in expected_frequency
        if int(count) > 0
    )
    assert int(op.workspace.num_valid_ids[0]) == expected_blocks * config.route_tile_m
    assert int(op.workspace.num_valid_ids[1]) == TOKENS
    _assert_close(actual, expected)

    # Kernel/JIT cache specialization must include tile_m.  Running 16 then 32
    # in one process catches accidental reuse of captured DSL constants.
    config_tile32 = replace(config, tile_m=32)
    op_tile32 = SonicMoE(config_tile32, prepared)
    frequency_tile32 = torch.empty_like(frequency)
    actual_tile32 = op_tile32.forward_routes(
        x,
        token_indices,
        expert_indices,
        route_weights,
        expert_frequency_out=frequency_tile32,
    )
    torch.cuda.synchronize()
    assert op_tile32.workspace is not None
    expected_tile32_blocks = sum(
        (int(count) + config_tile32.route_tile_m - 1) // config_tile32.route_tile_m
        for count in expected_frequency
        if int(count) > 0
    )
    assert int(op_tile32.workspace.num_valid_ids[0]) == expected_tile32_blocks * config_tile32.route_tile_m
    assert torch.equal(frequency_tile32, expected_frequency)
    _assert_close(actual_tile32, expected)

    actual_tile16_again = op.forward_routes(
        x,
        token_indices,
        expert_indices,
        route_weights,
    )
    torch.cuda.synchronize()
    assert op.workspace is not None
    assert int(op.workspace.num_valid_ids[0]) == expected_blocks * config.route_tile_m
    assert torch.equal(op.workspace.expert_frequency, expected_frequency)
    _assert_close(actual_tile16_again, expected)

    empty_indices = torch.empty(0, dtype=torch.int32, device=x.device)
    empty_weights = torch.empty(0, dtype=torch.float32, device=x.device)
    empty_frequency = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=x.device)
    empty_actual = op.forward_routes(
        x,
        empty_indices,
        empty_indices,
        empty_weights,
        expert_frequency_out=empty_frequency,
    )
    torch.cuda.synchronize()
    assert torch.count_nonzero(empty_actual) == 0
    assert torch.count_nonzero(empty_frequency) == 0
    assert op.workspace is not None and op.workspace.max_m_blocks == 0


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_sonic_moe_ragged_high_fan_in_matches_fp32_reference(dtype):
    """An up-rounded token may receive many experts and A16 atomic contributions."""

    device = _gfx950_device()
    tokens, experts = 3, 64
    config = _config(
        num_experts=experts,
        top_k=1,
        compute_dtype="fp16" if dtype == torch.float16 else "bf16",
    )
    generator = torch.Generator(device=device).manual_seed(59)
    x = torch.randn((tokens, HIDDEN_SIZE), dtype=torch.float32, device=device, generator=generator).to(dtype)
    w1 = (
        torch.randn(
            (experts, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        / math.sqrt(HIDDEN_SIZE)
    ).to(dtype)
    w2 = (
        torch.randn(
            (experts, HIDDEN_SIZE, INTERMEDIATE_SIZE),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        / math.sqrt(INTERMEDIATE_SIZE)
    ).to(dtype)
    token_indices = torch.zeros(experts, dtype=torch.int32, device=device)
    expert_indices = torch.arange(experts, dtype=torch.int32, device=device)
    route_weights = torch.softmax(
        torch.randn(experts, dtype=torch.float32, device=device, generator=generator),
        dim=0,
    )
    frequency = torch.empty(experts, dtype=torch.int32, device=device)

    expected_row = torch.zeros(HIDDEN_SIZE, dtype=torch.float32, device=device)
    for expert in range(experts):
        gate_up = (w1[expert].float() @ x[0].float()).to(dtype).float()
        gate, up = gate_up.split(INTERMEDIATE_SIZE)
        activated = (torch.nn.functional.silu(gate) * up).to(dtype).float()
        projected = (w2[expert].float() @ activated).to(dtype).float()
        expected_row.add_(projected * route_weights[expert])
    expected = torch.zeros_like(x)
    expected[0] = expected_row.to(dtype)

    op = SonicMoE(config, _prepare_dense_weights(w1, w2, config))
    actual = op.forward_routes(
        x,
        token_indices,
        expert_indices,
        route_weights,
        expert_frequency_out=frequency,
    )
    torch.cuda.synchronize()

    assert torch.equal(frequency, torch.ones_like(frequency))
    assert torch.count_nonzero(actual[1:]) == 0
    _assert_close(actual, expected)


def test_sonic_moe_mxfp4_weight_only_forward():
    """A16W4 keeps BF16 activations while consuming per-1x32 MXFP4 weights."""

    device = _gfx950_device()
    config = SonicMoEConfig(
        hidden_size=256,
        intermediate_size=256,
        num_experts=4,
        top_k=2,
        tile_m=16,
        tile_n=256,
        tile_k=256,
    )
    generator = torch.Generator(device=device).manual_seed(43)
    x = torch.randn((7, 256), device=device, dtype=torch.bfloat16, generator=generator)
    w1 = (torch.randn((4, 512, 256), device=device, dtype=torch.float32, generator=generator) / math.sqrt(256)).to(
        torch.bfloat16
    )
    w2 = (torch.randn((4, 256, 256), device=device, dtype=torch.float32, generator=generator) / math.sqrt(256)).to(
        torch.bfloat16
    )
    logits = torch.randn((7, 4), device=device, dtype=torch.bfloat16, generator=generator)

    prepared = prepare_sonic_mxfp4_weights(w1, w2, config)
    assert prepared.weight_dtype == "mxfp4"
    assert prepared.gate_up.dtype == torch.uint8
    assert prepared.down.dtype == torch.uint8
    assert prepared.gate_up_scale is not None
    assert prepared.down_scale is not None
    assert prepared.gate_up.numel() * 2 == w1.numel()
    assert prepared.down.numel() * 2 == w2.numel()

    expected = sonic_moe_mxfp4_reference(x, w1, w2, logits, config)
    actual = SonicMoE(config, prepared)(x, logits)
    torch.cuda.synchronize()
    _assert_close(actual, expected)
    cosine = torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0).item()
    assert cosine >= 0.999


def test_sonic_moe_mxfp4_quantizer_uses_rne_midpoints():
    """Golden codes prevent a circular quantize/dequantize oracle."""

    from tests.kernels.utils import gemm_common_utils

    device = _gfx950_device()
    weight = torch.zeros((1, 3, 32), dtype=torch.float32, device=device)
    weight[0, 0, :4] = torch.tensor([0.75, 1.75, 3.5, 4.0], device=device)
    weight[0, 1, :4] = torch.tensor([-0.75, -1.75, -3.5, -4.0], device=device)
    weight[0, 2, 0] = -0.0
    packed, scale = _quantize_mxfp4_weight(weight)

    assert scale[0, 0, 0].item() == 127
    assert scale[0, 1, 0].item() == 127
    assert scale[0, 2, 0].item() == 0
    assert packed[0, 0, 0].item() == 0x42
    assert packed[0, 0, 1].item() == 0x66
    assert packed[0, 1, 0].item() == 0xCA
    assert packed[0, 1, 1].item() == 0xEE
    assert packed[0, 2, 0].item() == 0x08
    assert not packed[0, 2, 1:].any().item()

    blocks = weight.view(-1, 32)
    reference_scale = gemm_common_utils.f32_to_e8m0(blocks.abs().amax(dim=1) / 4.0).view(torch.uint8)
    reference_values = blocks / gemm_common_utils.e8m0_to_f32(reference_scale)[:, None]
    reference_packed = gemm_common_utils.f32_to_mxfp4(reference_values).view(torch.uint8).view_as(packed)
    assert torch.equal(scale.view(-1), reference_scale)
    assert torch.equal(packed, reference_packed)


def test_sonic_moe_reuses_workspace_by_device_stream_and_token_count():
    config = _config()
    x, w1, w2, router_logits = _make_case()
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))

    first = op(x, router_logits)
    torch.cuda.synchronize()
    workspace_t7 = op.workspace
    assert workspace_t7 is not None

    out = torch.empty_like(first)
    returned = op(x, router_logits, out=out)
    torch.cuda.synchronize()
    assert returned is out
    assert op.workspace is workspace_t7
    torch.testing.assert_close(out.float(), first.float(), rtol=3e-2, atol=5e-2)

    x_t3, _, _, logits_t3 = _make_case(tokens=3, seed=23)
    result_t3 = op(x_t3, logits_t3)
    torch.cuda.synchronize()
    workspace_t3 = op.workspace
    assert result_t3.shape == (3, HIDDEN_SIZE)
    assert workspace_t3 is not None
    assert workspace_t3 is not workspace_t7

    op(x, router_logits)
    torch.cuda.synchronize()
    assert op.workspace is workspace_t7


def test_sonic_moe_workspace_cache_is_bounded_lru():
    config = _config()
    _, w1, w2, _ = _make_case()
    op = SonicMoE(
        config,
        prepare_sonic_bf16_weights(w1, w2, config),
        max_cached_workspaces=2,
    )

    workspace_t7 = op.reserve(7)
    workspace_t3 = op.reserve(3)
    cached = tuple(op._workspaces.values())
    assert cached[0] is workspace_t7
    assert cached[1] is workspace_t3

    assert op.reserve(7) is workspace_t7
    cached = tuple(op._workspaces.values())
    assert cached[0] is workspace_t3
    assert cached[1] is workspace_t7

    workspace_t5 = op.reserve(5)
    cached = tuple(op._workspaces.values())
    assert cached[0] is workspace_t7
    assert cached[1] is workspace_t5

    replacement_t3 = op.reserve(3)
    cached = tuple(op._workspaces.values())
    assert replacement_t3 is not workspace_t3
    assert cached[0] is workspace_t5
    assert cached[1] is replacement_t3
    assert op.workspace is replacement_t3

    op.clear_workspace()
    assert not op._workspaces
    assert op.workspace is None


@pytest.mark.parametrize(
    ("capacity", "exception"),
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_sonic_moe_rejects_invalid_workspace_cache_capacity(capacity, exception):
    config = _config()
    _, w1, w2, _ = _make_case()
    weights = prepare_sonic_bf16_weights(w1, w2, config)

    with pytest.raises(exception, match="max_cached_workspaces"):
        SonicMoE(config, weights, max_cached_workspaces=capacity)


@pytest.mark.parametrize("entrypoint", ["router", "topk"])
def test_sonic_moe_serializes_same_workspace_enqueues(monkeypatch, entrypoint):
    config = _config()
    x, w1, w2, router_logits = _make_case()
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    stream = torch.cuda.Stream(device=x.device)
    with torch.cuda.device(x.device), torch.cuda.stream(stream):
        workspace = op.reserve(TOKENS)

    first_sort_entered = threading.Event()
    second_lock_attempted = threading.Event()
    second_sort_entered = threading.Event()
    release_first_sort = threading.Event()
    first_gemms_entered = threading.Event()
    release_first_gemms = threading.Event()
    order = []
    errors = []
    result_lock = threading.Lock()

    class TrackingLock:
        def __init__(self):
            self._lock = threading.Lock()

        def __enter__(self):
            if threading.current_thread().name == "sonic-second":
                second_lock_attempted.set()
            self._lock.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._lock.release()

    workspace._launch_lock = TrackingLock()

    def fake_sort(*args, **kwargs):
        del args, kwargs
        name = threading.current_thread().name
        with result_lock:
            order.append((name, "sort"))
        if name == "sonic-first":
            first_sort_entered.set()
            if not release_first_sort.wait(timeout=5):
                raise TimeoutError("first sort was not released")
        else:
            second_sort_entered.set()

    def fake_grouped_gemms(hidden_states, active_workspace, out):
        assert hidden_states is x
        assert active_workspace is workspace
        name = threading.current_thread().name
        with result_lock:
            order.append((name, "gemms"))
        if name == "sonic-first":
            first_gemms_entered.set()
            if not release_first_gemms.wait(timeout=5):
                raise TimeoutError("first GEMMs were not released")
        return out

    monkeypatch.setattr("kernels.moe.sonic.moe_softmax_sort_flydsl", fake_sort)
    monkeypatch.setattr("kernels.moe.sonic.moe_sorting_flydsl", fake_sort)
    monkeypatch.setattr(op, "_run_grouped_gemms", fake_grouped_gemms)

    outputs = (torch.empty_like(x), torch.empty_like(x))

    def run(output):
        try:
            with torch.cuda.device(x.device), torch.cuda.stream(stream):
                if entrypoint == "router":
                    result = op(x, router_logits, out=output)
                else:
                    result = op.forward_topk(
                        x,
                        topk_ids,
                        topk_weights,
                        out=output,
                    )
            assert result is output
        except Exception as error:
            with result_lock:
                errors.append(error)

    first = threading.Thread(target=run, args=(outputs[0],), name="sonic-first")
    second = threading.Thread(target=run, args=(outputs[1],), name="sonic-second")
    first.start()
    try:
        assert first_sort_entered.wait(timeout=5)
        second.start()
        assert second_lock_attempted.wait(timeout=5)
        assert not second_sort_entered.is_set()
        release_first_sort.set()
        assert first_gemms_entered.wait(timeout=5)
        assert not second_sort_entered.is_set()
        release_first_gemms.set()
    finally:
        release_first_sort.set()
        release_first_gemms.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not errors
    assert order == [
        ("sonic-first", "sort"),
        ("sonic-first", "gemms"),
        ("sonic-second", "sort"),
        ("sonic-second", "gemms"),
    ]


def test_sonic_moe_workspace_bound_scales_with_active_experts():
    """Decode must not allocate or launch one padded tile for every expert."""

    device = _gfx950_device()
    dense_routes = SonicMoEWorkspace.allocate(
        _config(num_experts=2, top_k=2, tile_m=16),
        tokens=16,
        device=device,
    )
    assert dense_routes.max_m_blocks == 2
    assert dense_routes.max_padded_tokens == 32

    split_tiles = SonicMoEWorkspace.allocate(
        _config(num_experts=2, top_k=2, tile_m=32, down_tile_m=128),
        tokens=16,
        device=device,
    )
    assert split_tiles.route_tile_m == 128
    assert split_tiles.max_m_blocks == 2
    assert split_tiles.max_padded_tokens == 256
    assert split_tiles.stage1_max_m_blocks == 8
    assert split_tiles.stage2_max_m_blocks == 2

    config = _config(num_experts=896, top_k=2, tile_m=32)
    generator = torch.Generator(device=device).manual_seed(53)
    x = torch.randn((1, HIDDEN_SIZE), dtype=torch.bfloat16, device=device, generator=generator)
    w1 = torch.randn(
        (896, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) / math.sqrt(HIDDEN_SIZE)
    w2 = torch.randn(
        (896, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ) / math.sqrt(INTERMEDIATE_SIZE)
    logits = torch.randn((1, 896), dtype=torch.bfloat16, device=device, generator=generator)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    expected = sonic_moe_reference(x, w1, w2, logits, config)
    actual = op(x, logits)
    torch.cuda.synchronize()

    workspace = op.workspace
    assert workspace is not None
    assert workspace.max_padded_tokens == 64
    assert workspace.max_m_blocks == 2
    assert workspace.intermediate.shape == (64, INTERMEDIATE_SIZE)
    _assert_close(actual, expected)

    topk_ids, topk_weights = _topk_from_logits(logits, config)
    ragged_frequency = torch.empty(896, dtype=torch.int32, device=device)
    ragged_actual = op.forward_routes(
        x,
        torch.zeros(config.top_k, dtype=torch.int32, device=device),
        topk_ids.reshape(-1),
        topk_weights.reshape(-1),
        expert_frequency_out=ragged_frequency,
    )
    torch.cuda.synchronize()
    assert op.workspace is not None
    assert op.workspace.routes == config.top_k
    assert op.workspace.max_padded_tokens == 64
    assert int(ragged_frequency.sum()) == config.top_k
    _assert_close(ragged_actual, expected)

    mxfp4_weights = prepare_sonic_mxfp4_weights(w1, w2, config)
    mxfp4_expected = sonic_moe_mxfp4_reference(x, w1, w2, logits, config)
    mxfp4_op = SonicMoE(config, mxfp4_weights)
    mxfp4_actual = mxfp4_op(x, logits)
    torch.cuda.synchronize()
    assert mxfp4_op.workspace is not None
    assert mxfp4_op.workspace.max_padded_tokens == 64
    _assert_close(mxfp4_actual, mxfp4_expected)


def test_sonic_moe_autotuner_search_and_disk_cache(tmp_path):
    config = _config()
    x, w1, w2, router_logits = _make_case(seed=47)
    weights = prepare_sonic_bf16_weights(w1, w2, config)
    candidates = (config, replace(config, tile_m=32))
    with pytest.raises(ValueError, match="at least one"):
        SonicMoEAutotuner(config, weights, candidates=[], cache_dir=tmp_path)
    tuner = SonicMoEAutotuner(
        config,
        weights,
        candidates=candidates,
        warmup=0,
        rep=1,
        cache_dir=tmp_path,
    )
    b1 = torch.zeros(
        (NUM_EXPERTS, config.stage1_projection_size),
        dtype=torch.bfloat16,
        device=x.device,
    )
    b2 = torch.zeros((NUM_EXPERTS, HIDDEN_SIZE), dtype=torch.bfloat16, device=x.device)
    biased_tuner = SonicMoEAutotuner(
        config,
        prepare_sonic_bf16_weights(w1, w2, config, b1=b1, b2=b2),
        candidates=candidates,
        warmup=0,
        rep=1,
        cache_dir=tmp_path / "biased",
    )
    assert tuner._cache_key(x, router_logits) != biased_tuner._cache_key(x, router_logits)
    reduce_config = replace(config, stage2_output_mode="reduce")
    reduce_tuner = SonicMoEAutotuner(
        reduce_config,
        weights,
        candidates=(reduce_config,),
        warmup=0,
        rep=1,
        cache_dir=tmp_path / "reduce",
    )
    assert tuner._cache_key(x, router_logits) != reduce_tuner._cache_key(x, router_logits)
    split_m_config = replace(config, down_tile_m=128)
    split_m_tuner = SonicMoEAutotuner(
        split_m_config,
        weights,
        candidates=(split_m_config,),
        warmup=0,
        rep=1,
        cache_dir=tmp_path / "split-m",
    )
    assert tuner._cache_key(x, router_logits) != split_m_tuner._cache_key(x, router_logits)
    k_wave_config = replace(config, stage1_k_wave=2)
    k_wave_tuner = SonicMoEAutotuner(
        k_wave_config,
        weights,
        candidates=(k_wave_config,),
        warmup=0,
        rep=1,
        cache_dir=tmp_path / "stage1-k-wave",
    )
    assert tuner._cache_key(x, router_logits) != k_wave_tuner._cache_key(x, router_logits)
    with pytest.raises(ValueError, match="semantic values differ"):
        SonicMoEAutotuner(
            reduce_config,
            weights,
            candidates=(config,),
            warmup=0,
            rep=1,
            cache_dir=tmp_path / "mixed-output-mode",
        )

    expected = sonic_moe_reference(x, w1, w2, router_logits, config)
    actual = tuner(x, router_logits)
    torch.cuda.synchronize()
    _assert_close(actual, expected)
    assert tuner.best_config in candidates
    assert tuner.search_count == 1
    assert tuner.cache_file.is_file()

    tuner(x, router_logits)
    torch.cuda.synchronize()
    assert tuner.search_count == 1

    reloaded = SonicMoEAutotuner(
        config,
        weights,
        candidates=candidates,
        warmup=0,
        rep=1,
        cache_dir=tmp_path,
    )
    cached = reloaded(x, router_logits)
    torch.cuda.synchronize()
    _assert_close(cached, expected)
    assert reloaded.search_count == 0
    assert reloaded.best_config == tuner.best_config
    assert len(reloaded._ops) == 1

    scaled = torch.randn_like(actual.float())
    assert not SonicMoEAutotuner._candidate_matches(scaled, scaled * 10)

    profiled = SonicMoEAutotuner(
        config,
        weights,
        candidates=candidates,
        warmup=0,
        rep=1,
        cache_dir=tmp_path,
        profile_key="decode-skew",
    )
    profiled(x, router_logits)
    torch.cuda.synchronize()
    assert profiled.search_count == 1

    unvalidated = SonicMoEAutotuner(
        config,
        weights,
        candidates=(config,),
        warmup=0,
        rep=1,
        cache_dir=tmp_path / "unvalidated",
        validate_candidates=False,
    )
    unvalidated(x, router_logits)
    torch.cuda.synchronize()
    assert unvalidated.search_count == 1
    assert not unvalidated.cache_file.exists()

    non_object_cache_dir = tmp_path / "non_object_json"
    non_object_cache_dir.mkdir()
    non_object_cache_file = non_object_cache_dir / "sonic_moe.json"
    non_object_cache_file.write_text("[]", encoding="utf-8")
    recovered = SonicMoEAutotuner(
        config,
        weights,
        candidates=(config,),
        warmup=0,
        rep=1,
        cache_dir=non_object_cache_dir,
    )
    recovered(x, router_logits)
    torch.cuda.synchronize()
    assert recovered.search_count == 1
    rewritten_cache = non_object_cache_file.read_text(encoding="utf-8")
    assert '"version": 16' in rewritten_cache
    assert '"stage1_k_wave": 1' in rewritten_cache


def test_sonic_moe_autotune_cache_separates_exact_pipeline_policy(monkeypatch, tmp_path):
    """Keep exact-M generated-kernel policy separate inside one M bucket."""

    config = _config()
    _, w1, w2, router_logits = _make_case(seed=257)
    tuner = SonicMoEAutotuner(
        config,
        prepare_sonic_bf16_weights(w1, w2, config),
        candidates=(config,),
        warmup=0,
        rep=1,
        cache_dir=tmp_path,
    )
    monkeypatch.setattr(
        "kernels.moe.sonic_autotune._stage2_stages",
        lambda _config, tokens: 2 if tokens == 4096 else 1,
    )
    x_4095 = torch.empty((4095, HIDDEN_SIZE), dtype=torch.bfloat16, device=w1.device)
    x_4096 = torch.empty((4096, HIDDEN_SIZE), dtype=torch.bfloat16, device=w1.device)

    identity_4095 = json.loads(tuner._cache_key(x_4095, router_logits))
    identity_4096 = json.loads(tuner._cache_key(x_4096, router_logits))

    assert identity_4095["tokens_bucket"] == identity_4096["tokens_bucket"] == 4096
    assert identity_4095["stage2_pipeline_stages"] == [1]
    assert identity_4096["stage2_pipeline_stages"] == [2]
    assert identity_4095 != identity_4096


def test_sonic_moe_router_and_multiphase_sort_fallback():
    """T > 128 exercises supplied top-k scratch and sorting workspace."""

    config = _config()
    x, w1, w2, router_logits = _make_case(tokens=129, seed=31)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))

    expected = sonic_moe_reference(x, w1, w2, router_logits, config)
    actual = op(x, router_logits)
    torch.cuda.synchronize()
    assert op.workspace is not None
    assert op.workspace.sorting_workspace is not None
    _assert_close(actual, expected)


def test_sonic_moe_fused_router_handles_negative_infinity_masks():
    """Fewer than K finite logits must not emit an invalid expert index."""

    device = _gfx950_device()
    config = _config(num_experts=8, top_k=4)
    generator = torch.Generator(device=device).manual_seed(173)
    x = torch.randn((1, HIDDEN_SIZE), device=device, dtype=torch.bfloat16, generator=generator)
    w1 = (
        torch.randn(
            (config.num_experts, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        / math.sqrt(HIDDEN_SIZE)
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(
            (config.num_experts, HIDDEN_SIZE, INTERMEDIATE_SIZE),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        / math.sqrt(INTERMEDIATE_SIZE)
    ).to(torch.bfloat16)
    logits = torch.full(
        (1, config.num_experts),
        float("-inf"),
        device=device,
        dtype=torch.bfloat16,
    )
    logits[0, 5] = 2.0
    logits[0, 2] = 1.0
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))

    expected = sonic_moe_reference(x, w1, w2, logits, config)
    actual = op(x, logits)
    torch.cuda.synchronize()

    assert op.workspace is not None
    route_blocks = int(op.workspace.num_valid_ids[0].item()) // config.route_tile_m
    selected_experts = op.workspace.sorted_expert_ids[:route_blocks]
    assert torch.all((selected_experts >= 0) & (selected_experts < config.num_experts))
    assert set(selected_experts.cpu().tolist()) == {0, 1, 2, 5}
    assert torch.isfinite(actual).all()
    _assert_close(actual, expected)


@pytest.mark.parametrize(
    ("compute_dtype", "torch_dtype", "renormalize"),
    (
        ("bf16", torch.bfloat16, True),
        ("fp16", torch.float16, False),
    ),
)
@pytest.mark.parametrize("stage2_output_mode", ("atomic", "reduce"))
def test_sonic_moe_single_token_direct_router_preserves_route_slots(
    compute_dtype,
    torch_dtype,
    renormalize,
    stage2_output_mode,
):
    """The T=1 fast path emits one padded block for every top-k slot."""

    config = _config(
        stage2_output_mode=stage2_output_mode,
        compute_dtype=compute_dtype,
        renormalize=renormalize,
    )
    x, w1, w2, router_logits = _make_case(tokens=1, seed=181, dtype=torch_dtype)
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    op = SonicMoE(config, _prepare_dense_weights(w1, w2, config))

    expected = sonic_moe_reference(x, w1, w2, router_logits, config)
    actual = op(x, router_logits)
    torch.cuda.synchronize()

    workspace = op.workspace
    assert workspace is not None
    assert int(workspace.num_valid_ids[0]) == config.top_k * config.route_tile_m
    assert int(workspace.num_valid_ids[1]) == 1
    block_starts = torch.arange(config.top_k, device=x.device) * config.route_tile_m
    packed = workspace.sorted_token_ids[block_starts]
    assert torch.equal(packed & 0x00FFFFFF, torch.zeros_like(packed))
    assert torch.equal(packed >> 24, torch.arange(config.top_k, dtype=torch.int32, device=x.device))
    assert torch.equal(workspace.sorted_expert_ids[: config.top_k], topk_ids[0])
    torch.testing.assert_close(
        workspace.sorted_weights[block_starts],
        topk_weights[0],
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.count_nonzero(workspace.sorted_weights) == config.top_k
    _assert_close(actual, expected)

    topk_out = torch.empty_like(x)
    returned = op.forward_topk(x, topk_ids, topk_weights, out=topk_out)
    torch.cuda.synchronize()
    assert returned is topk_out
    assert op.workspace is workspace
    assert torch.equal(workspace.sorted_expert_ids[: config.top_k], topk_ids[0])
    assert torch.equal(
        workspace.sorted_token_ids[block_starts] >> 24,
        torch.arange(config.top_k, dtype=torch.int32, device=x.device),
    )
    torch.testing.assert_close(
        workspace.sorted_weights[block_starts],
        topk_weights[0],
        rtol=1e-5,
        atol=1e-6,
    )
    _assert_close(topk_out, expected)


@pytest.mark.parametrize("tokens", (1, 7, 129), ids=("direct-t1", "oneshot-sort", "multiphase-sort"))
@pytest.mark.parametrize("renormalize", (True, False), ids=("renorm", "full-softmax"))
def test_sonic_moe_logits_frequency_matches_selected_routes(tokens, renormalize):
    """The opt-in logits API reports exact frequencies on every sort path."""

    config = _config(renormalize=renormalize)
    x, w1, w2, router_logits = _make_case(tokens=tokens, seed=211 + tokens)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    expected = sonic_moe_reference(x, w1, w2, router_logits, config)
    topk_ids, _ = _topk_from_logits(router_logits, config)
    expected_frequency = torch.bincount(
        topk_ids.reshape(-1).to(torch.int64),
        minlength=config.num_experts,
    ).to(torch.int32)
    frequency = torch.full(
        (config.num_experts,),
        -17,
        dtype=torch.int32,
        device=x.device,
    )
    out = torch.empty_like(x)
    x_before = x.clone()
    logits_before = router_logits.clone()

    returned = op(
        x,
        router_logits,
        out=out,
        expert_frequency_out=frequency,
    )
    torch.cuda.synchronize()

    assert returned is out
    assert torch.equal(frequency, expected_frequency)
    assert int(frequency.sum()) == tokens * config.top_k
    assert torch.equal(x, x_before)
    assert torch.equal(router_logits, logits_before)
    _assert_close(out, expected)


@pytest.mark.parametrize("renormalize", (True, False), ids=("renorm", "full-softmax"))
def test_sonic_moe_prevalidated_logits_matches_public_path(renormalize):
    """The adapter-only entry preserves ownership and public-path results."""

    config = _config(renormalize=renormalize)
    x, w1, w2, router_logits = _make_case(tokens=1, seed=227)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    public_out = torch.empty_like(x)
    trusted_out = torch.empty_like(x)
    public_frequency = torch.empty(
        config.num_experts,
        dtype=torch.int32,
        device=x.device,
    )
    trusted_frequency = torch.empty_like(public_frequency)

    public_result = op(
        x,
        router_logits,
        out=public_out,
        expert_frequency_out=public_frequency,
    )
    trusted_result = op._forward_from_logits_prevalidated(
        x,
        router_logits,
        trusted_out,
        trusted_frequency,
    )
    torch.cuda.synchronize()

    assert public_result is public_out
    assert trusted_result is trusted_out
    assert torch.equal(trusted_frequency, public_frequency)
    _assert_close(trusted_out, public_out)


def test_sonic_moe_public_call_validates_before_prevalidated_launch(monkeypatch):
    """Extracting the trusted launcher must not weaken the public contract."""

    config = _config()
    x, w1, w2, router_logits = _make_case(tokens=1, seed=228)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))

    def launch_must_not_run(*args, **kwargs):
        raise AssertionError("invalid public inputs must be rejected before launch")

    monkeypatch.setattr(op, "_launch_prevalidated_logits", launch_must_not_run)
    with pytest.raises(TypeError, match="hidden_states must use"):
        op(x.float(), router_logits)
    with pytest.raises(ValueError, match="router_logits must have shape"):
        op(x, router_logits[:, :-1])


def test_sonic_moe_logits_frequency_is_fully_overwritten_between_calls():
    config = _config()
    x, w1, w2, router_logits = _make_case(tokens=1, seed=229)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    frequency = torch.full(
        (config.num_experts,),
        12345,
        dtype=torch.int32,
        device=x.device,
    )

    op(x, router_logits, expert_frequency_out=frequency)
    torch.cuda.synchronize()
    first = frequency.clone()
    first_ids, _ = _topk_from_logits(router_logits, config)
    first_expected = torch.bincount(first_ids.flatten().long(), minlength=config.num_experts).to(torch.int32)
    assert torch.equal(first, first_expected)

    second_logits = torch.full_like(router_logits, -10)
    second_logits[0, 1] = 4
    second_logits[0, 3] = 3
    frequency.fill_(6789)
    op(x, second_logits, expert_frequency_out=frequency)
    torch.cuda.synchronize()
    assert torch.equal(
        frequency,
        torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=x.device),
    )


def test_sonic_moe_logits_frequency_matches_torch_router_fallback():
    """Expert counts also work when E has no exact FlyDSL router layout."""

    device = _gfx950_device()
    experts = 17
    config = _config(num_experts=experts)
    assert not config.supports_flydsl_router
    generator = torch.Generator(device=device).manual_seed(231)
    x = torch.randn((3, HIDDEN_SIZE), dtype=torch.bfloat16, device=device, generator=generator)
    w1 = torch.randn(
        (experts, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w2 = torch.randn(
        (experts, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    logits = torch.randn((3, experts), dtype=torch.bfloat16, device=device, generator=generator)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    frequency = torch.empty(experts, dtype=torch.int32, device=device)

    op(x, logits, expert_frequency_out=frequency)
    torch.cuda.synchronize()

    expected_ids, _ = _topk_from_logits(logits, config)
    expected = torch.bincount(expected_ids.flatten().long(), minlength=experts).to(torch.int32)
    assert torch.equal(frequency, expected)


@pytest.mark.parametrize("renormalize", (True, False), ids=("renorm", "full-softmax"))
def test_sonic_moe_nonfinite_logits_keep_frequency_ids_in_range(renormalize):
    config = _config(renormalize=renormalize)
    x, w1, w2, router_logits = _make_case(tokens=1, seed=233)
    router_logits[0, 1] = float("nan")
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    frequency = torch.empty(config.num_experts, dtype=torch.int32, device=x.device)

    op(x, router_logits, expert_frequency_out=frequency)
    torch.cuda.synchronize()

    assert torch.all(frequency >= 0)
    assert torch.all(frequency <= 1)
    assert int(frequency.sum()) == config.top_k


def test_sonic_moe_logits_frequency_validation_and_aliasing():
    config = _config()
    x, w1, w2, router_logits = _make_case(tokens=7, seed=239)
    prepared = prepare_sonic_bf16_weights(w1, w2, config)
    op = SonicMoE(config, prepared)

    with pytest.raises(TypeError, match="torch.Tensor"):
        op(x, router_logits, expert_frequency_out=[0] * config.num_experts)
    with pytest.raises(ValueError, match="shape"):
        op(
            x,
            router_logits,
            expert_frequency_out=torch.empty(config.num_experts - 1, dtype=torch.int32, device=x.device),
        )
    with pytest.raises(ValueError, match="contiguous int32"):
        op(
            x,
            router_logits,
            expert_frequency_out=torch.empty(config.num_experts, dtype=torch.int64, device=x.device),
        )
    with pytest.raises(ValueError, match="same ROCm device"):
        op(
            x,
            router_logits,
            expert_frequency_out=torch.empty(config.num_experts, dtype=torch.int32),
        )
    noncontiguous = torch.empty(config.num_experts * 2, dtype=torch.int32, device=x.device)[::2]
    assert not noncontiguous.is_contiguous()
    with pytest.raises(ValueError, match="contiguous int32"):
        op(x, router_logits, expert_frequency_out=noncontiguous)

    input_alias = router_logits.view(torch.int32).flatten()[: config.num_experts]
    assert tuple(input_alias.shape) == (config.num_experts,)
    with pytest.raises(ValueError, match="must not alias an input or output"):
        op(x, router_logits, expert_frequency_out=input_alias)

    out = torch.empty_like(x)
    output_alias = out.view(torch.int32).flatten()[: config.num_experts]
    with pytest.raises(ValueError, match="must not alias an input or output"):
        op(x, router_logits, out=out, expert_frequency_out=output_alias)

    workspace = op.reserve(x.shape[0])
    with pytest.raises(ValueError, match="must not alias internal workspace storage"):
        op(x, router_logits, expert_frequency_out=workspace.expert_frequency)


def test_sonic_moe_uses_independent_workspaces_across_streams():
    config = _config()
    x_a, w1, w2, logits_a = _make_case(seed=37)
    x_b, _, _, logits_b = _make_case(seed=41)
    op = SonicMoE(
        config,
        prepare_sonic_bf16_weights(w1, w2, config),
        max_cached_workspaces=1,
    )
    expected_a = sonic_moe_reference(x_a, w1, w2, logits_a, config)
    expected_b = sonic_moe_reference(x_b, w1, w2, logits_b, config)
    out_a = torch.empty_like(expected_a)
    out_b = torch.empty_like(expected_b)
    out_a_again = torch.empty_like(expected_a)

    default_stream = torch.cuda.current_stream(x_a.device)
    stream_a = torch.cuda.Stream(device=x_a.device)
    stream_b = torch.cuda.Stream(device=x_a.device)
    stream_a.wait_stream(default_stream)
    stream_b.wait_stream(default_stream)

    with torch.cuda.stream(stream_a):
        actual_a = op(x_a, logits_a, out=out_a)
        workspace_a_ref = weakref.ref(op.workspace)
    with torch.cuda.stream(stream_b):
        actual_b = op(x_b, logits_b, out=out_b)
        workspace_b_ref = weakref.ref(op.workspace)
    assert workspace_a_ref() is None
    with torch.cuda.stream(stream_a):
        actual_a_again = op(x_a, logits_a, out=out_a_again)
        workspace_a_again = op.workspace
    assert workspace_b_ref() is None

    torch.cuda.synchronize()
    assert actual_a is out_a
    assert actual_b is out_b
    assert actual_a_again is out_a_again
    assert workspace_a_again is not None
    assert len(op._workspaces) == 1
    assert next(iter(op._workspaces.values())) is workspace_a_again
    _assert_close(actual_a, expected_a)
    _assert_close(actual_b, expected_b)
    _assert_close(actual_a_again, expected_a)


def test_sonic_moe_gfx950_stage2_pipeline_public_forward_and_mixed_streams(monkeypatch):
    """Exercise the exact production pipeline cell beside its serial neighbor."""

    import kernels.moe.sonic as sonic_module

    device = _gfx950_device()
    config = SonicMoEConfig(
        hidden_size=4096,
        intermediate_size=2048,
        num_experts=64,
        top_k=8,
        tile_m=128,
        tile_n=256,
        tile_k=64,
        down_tile_m=128,
        down_tile_n=128,
        down_tile_k=64,
        stage2_xcd_swizzle=8,
    )
    generator = torch.Generator(device=device).manual_seed(263)
    x_4096 = torch.randn((4096, 4096), dtype=torch.bfloat16, device=device, generator=generator)
    logits_4096 = torch.randn((4096, 64), dtype=torch.bfloat16, device=device, generator=generator)
    w1 = torch.randn((64, 4096, 4096), dtype=torch.bfloat16, device=device, generator=generator)
    w2 = torch.randn((64, 4096, 2048), dtype=torch.bfloat16, device=device, generator=generator)
    w1.mul_(1.0 / math.sqrt(4096))
    w2.mul_(1.0 / math.sqrt(2048))
    prepared = prepare_sonic_bf16_weights(w1, w2, config)
    del w1, w2

    op = SonicMoE(config, prepared, max_cached_workspaces=4)
    x_2048 = x_4096[:2048]
    logits_2048 = logits_4096[:2048]
    serial_4096 = torch.empty_like(x_4096)
    serial_2048 = torch.empty_like(x_2048)

    # Establish independent public-forward baselines with the historical
    # serial Stage-2 loop, while retaining the identical router and Stage 1.
    with monkeypatch.context() as serial_patch:
        serial_patch.setattr(sonic_module, "_stage2_stages", lambda _config, _tokens: 1)
        op(x_4096, logits_4096, out=serial_4096)
        op(x_2048, logits_2048, out=serial_2048)
        torch.cuda.synchronize()

    selected_stages = []
    get_stage2_launcher = sonic_module._get_stage2_launcher

    def tracking_get_stage2_launcher(config, b_cache_mod, weight_dtype, has_bias, output_mode, stages, device_index):
        selected_stages.append((int(stages), int(torch.cuda.current_stream(device).cuda_stream)))
        return get_stage2_launcher(
            config,
            b_cache_mod,
            weight_dtype,
            has_bias,
            output_mode,
            stages,
            device_index,
        )

    monkeypatch.setattr(sonic_module, "_get_stage2_launcher", tracking_get_stage2_launcher)
    pipeline_4096 = torch.empty_like(x_4096)
    production_2048 = torch.empty_like(x_2048)
    default_stream = torch.cuda.current_stream(device)
    stream_4096 = torch.cuda.Stream(device=device)
    stream_2048 = torch.cuda.Stream(device=device)
    stream_4096.wait_stream(default_stream)
    stream_2048.wait_stream(default_stream)

    with torch.cuda.stream(stream_4096):
        assert op(x_4096, logits_4096, out=pipeline_4096) is pipeline_4096
        workspace_4096 = op.workspace
    with torch.cuda.stream(stream_2048):
        assert op(x_2048, logits_2048, out=production_2048) is production_2048
        workspace_2048 = op.workspace
    torch.cuda.synchronize()

    assert workspace_4096 is not None and workspace_4096.routes is None
    assert workspace_2048 is not None and workspace_2048.routes is None
    assert workspace_4096 is not workspace_2048
    assert workspace_4096.tokens == 4096
    assert workspace_2048.tokens == 2048
    assert [stages for stages, _stream in selected_stages] == [2, 1]
    assert selected_stages[0][1] != selected_stages[1][1]
    _assert_close(pipeline_4096, serial_4096)
    _assert_close(production_2048, serial_2048)


@pytest.mark.multi_gpu
def test_sonic_moe_selects_input_device_and_restores_current_device():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least two ROCm GPUs")

    for index in (0, 1):
        device = torch.device("cuda", index)
        arch = str(getattr(torch.cuda.get_device_properties(device), "gcnArchName", ""))
        if not arch.startswith("gfx950"):
            pytest.skip(f"SonicMoE test requires two gfx950 devices, found {device}={arch!r}")

    target_device = torch.device("cuda", 1)
    with torch.cuda.device(target_device):
        config = _config()
        x, w1, w2, router_logits = _make_case(seed=71)
        assert x.device == target_device
        topk_ids, topk_weights = _topk_from_logits(router_logits, config)
        expected = sonic_moe_reference(x, w1, w2, router_logits, config)
        op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
        router_out = torch.empty_like(expected)
        topk_out = torch.empty_like(expected)

    with torch.cuda.device(0):
        assert torch.cuda.current_device() == 0
        actual_router = op(x, router_logits, out=router_out)
        assert torch.cuda.current_device() == 0
        actual_topk = op.forward_topk(x, topk_ids, topk_weights, out=topk_out)
        assert torch.cuda.current_device() == 0

    torch.cuda.synchronize(target_device)
    assert actual_router is router_out
    assert actual_topk is topk_out
    _assert_close(actual_router, expected)
    _assert_close(actual_topk, expected)


@pytest.mark.multi_gpu
def test_sonic_moe_runs_sequentially_on_two_devices():
    """A launcher materialized on cuda:0 must not be reused on cuda:1."""

    from flydsl.compiler.jit_function import _current_device_cache_signature
    from flydsl.runtime.device_runtime import get_device_runtime

    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least two ROCm GPUs")

    devices = (torch.device("cuda", 0), torch.device("cuda", 1))
    for device in devices:
        properties = torch.cuda.get_device_properties(device)
        arch = str(getattr(properties, "gcnArchName", ""))
        if not arch.startswith("gfx950"):
            pytest.skip(f"SonicMoE test requires two gfx950 devices, found {device}={arch!r}")

    config = _config()
    results = []
    _get_stage1_launcher.cache_clear()
    _get_stage2_launcher.cache_clear()
    try:
        for device, seed in zip(devices, (61, 67)):
            with torch.cuda.device(device):
                assert get_device_runtime().current_device_id() == device.index
                assert _current_device_cache_signature() == ("rocm", device.index)
                x, w1, w2, router_logits = _make_case(seed=seed)
                assert x.device == device

                expected = sonic_moe_reference(x, w1, w2, router_logits, config)
                actual = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))(x, router_logits)
                torch.cuda.synchronize(device)
                results.append((actual, expected))

        assert _get_stage1_launcher.cache_info().currsize == 2
        assert _get_stage2_launcher.cache_info().currsize == 2
    finally:
        _get_stage1_launcher.cache_clear()
        _get_stage2_launcher.cache_clear()

    for actual, expected in results:
        _assert_close(actual, expected)


def test_sonic_moe_unsupported_router_layout_falls_back():
    """Expert counts without an exact lane layout retain the torch fallback."""

    device = _gfx950_device()
    config = _config(num_experts=17)
    assert not config.supports_flydsl_router
    generator = torch.Generator(device=device).manual_seed(29)
    x = torch.randn((3, HIDDEN_SIZE), device=device, dtype=torch.bfloat16, generator=generator)
    w1 = torch.randn(
        (17, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ) / math.sqrt(HIDDEN_SIZE)
    w2 = torch.randn(
        (17, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ) / math.sqrt(INTERMEDIATE_SIZE)
    logits = torch.randn((3, 17), device=device, dtype=torch.bfloat16, generator=generator)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))

    expected = sonic_moe_reference(x, w1, w2, logits, config)
    actual = op(x, logits)
    torch.cuda.synchronize()
    _assert_close(actual, expected)


def test_sonic_moe_stage1_k_wave_config_and_default_candidates():
    config = _config()
    assert config.stage1_k_wave == 1

    k_wave2 = replace(config, stage1_k_wave=2)
    assert k_wave2.stage1_k_wave == 2
    candidate_k_waves = {
        candidate.stage1_k_wave
        for candidate in default_sonic_moe_candidates(k_wave2, "bf16")
    }
    assert {1, 2}.issubset(candidate_k_waves)

    k_wave4 = _config(hidden_size=512, stage1_k_wave=4)
    assert k_wave4.stage1_k_wave == 4
    for invalid in (0, 3, 8):
        with pytest.raises(ValueError, match="stage1_k_wave must be 1, 2, or 4"):
            _config(stage1_k_wave=invalid)
    with pytest.raises(ValueError, match=r"stage1_k_wave \* tile_k"):
        _config(hidden_size=384, stage1_k_wave=2)
    with pytest.raises(ValueError, match="stage1 tile needs"):
        SonicMoEConfig(
            hidden_size=512,
            intermediate_size=256,
            num_experts=4,
            top_k=2,
            tile_m=128,
            tile_n=256,
            tile_k=128,
            stage1_k_wave=4,
        )


@pytest.mark.parametrize("weight_dtype", ("bf16", "fp16"))
def test_sonic_moe_default_candidates_include_curated_gfx950_dense_profiles(weight_dtype):
    config = SonicMoEConfig(
        hidden_size=4096,
        intermediate_size=2048,
        num_experts=64,
        top_k=8,
    )
    candidates = default_sonic_moe_candidates(config, weight_dtype)

    assert len(candidates) <= 19
    assert len(candidates) == len(set(candidates))
    assert {
        (candidate.tile_m, candidate.stage2_tile_m)
        for candidate in candidates
    }.issuperset({(16, 16), (32, 128), (64, 128), (128, 128)})
    assert any(
        candidate.tile_m == 128
        and candidate.stage2_tile_m == 128
        and candidate.tile_n == 256
        and candidate.tile_k == 64
        and candidate.stage2_tile_n == 128
        and candidate.stage2_tile_k == 64
        and candidate.stage1_xcd_swizzle == 0
        and candidate.stage2_xcd_swizzle == 8
        for candidate in candidates
    )
    decode_k_waves = {
        candidate.stage1_k_wave
        for candidate in candidates
        if candidate.tile_m == 16 and candidate.stage2_tile_m == 16
    }
    assert {2, 4}.issubset(decode_k_waves)
    assert any(
        candidate.tile_m == 16
        and candidate.tile_n == 64
        and candidate.stage1_k_wave == 2
        and candidate.stage2_tile_m == 16
        and candidate.stage2_tile_n == 128
        and candidate.stage2_xcd_swizzle == 1
        for candidate in candidates
    )
    assert any(
        candidate.tile_m == 16
        and candidate.tile_n == 128
        and candidate.stage1_k_wave == 4
        and candidate.stage2_tile_m == 16
        and candidate.stage2_tile_n == 64
        and candidate.stage2_xcd_swizzle == 1
        for candidate in candidates
    )


@pytest.mark.parametrize(
    ("config", "expected"),
    (
        (
            SonicMoEConfig(
                hidden_size=2048,
                intermediate_size=768,
                num_experts=128,
                top_k=8,
            ),
            (128, 192, 64, 64, 256, 64, 8, 0, None, True, True),
        ),
        (
            SonicMoEConfig(
                hidden_size=4096,
                intermediate_size=14336,
                num_experts=8,
                top_k=2,
            ),
            (128, 256, 64, 128, 128, 64, 8, 8, None, True, True),
        ),
    ),
    ids=("e128-prefill", "e8-prefill"),
)
def test_sonic_moe_default_candidates_include_measured_prefill_profiles(config, expected):
    candidates = default_sonic_moe_candidates(config, "bf16")

    assert any(
        (
            candidate.tile_m,
            candidate.tile_n,
            candidate.tile_k,
            candidate.stage2_tile_m,
            candidate.stage2_tile_n,
            candidate.stage2_tile_k,
            candidate.stage1_xcd_swizzle,
            candidate.stage2_xcd_swizzle,
            candidate.stage2_pipeline_stages,
            candidate.stage1_write_padded_rows,
            candidate.stage1_lds_swizzle,
        )
        == expected
        for candidate in candidates
    )


@pytest.mark.parametrize("weight_dtype", (None, "mxfp4", "int4"))
def test_sonic_moe_default_candidates_keep_packed_weights_at_k128(weight_dtype):
    config = SonicMoEConfig(
        hidden_size=4096,
        intermediate_size=2048,
        num_experts=64,
        top_k=8,
    )
    candidates = default_sonic_moe_candidates(config, weight_dtype)

    assert candidates
    assert all(candidate.tile_k >= 128 for candidate in candidates)
    assert all(candidate.stage2_tile_k >= 128 for candidate in candidates)
    assert any(candidate.stage2_xcd_swizzle == 8 for candidate in candidates)
    with pytest.raises(TypeError, match="weight_dtype must be str or None"):
        default_sonic_moe_candidates(config, torch.bfloat16)


def test_sonic_moe_candidate_fingerprint_covers_gfx950_tuning_axes():
    config = SonicMoEConfig(
        hidden_size=4096,
        intermediate_size=2048,
        num_experts=64,
        top_k=8,
    )
    probes = (
        config,
        replace(config, down_tile_m=128),
        replace(config, stage1_k_wave=2),
        replace(config, stage2_xcd_swizzle=8),
        replace(config, tile_k=64),
        replace(config, down_tile_k=64),
        replace(config, stage1_lds_swizzle=True),
    )
    tuner = object.__new__(SonicMoEAutotuner)
    tuner.candidates = probes
    fingerprints = tuner._candidate_fingerprint()

    assert len({tuple(sorted(fingerprint.items())) for fingerprint in fingerprints}) == len(probes)
    assert fingerprints[1]["down_tile_m"] == 128
    assert fingerprints[2]["stage1_k_wave"] == 2
    assert fingerprints[3]["stage2_xcd_swizzle"] == 8
    assert fingerprints[4]["tile_k"] == 64
    assert fingerprints[5]["down_tile_k"] == 64
    assert fingerprints[6]["stage1_lds_swizzle"] is True


def test_sonic_moe_config_validation():
    device = _gfx950_device()
    config = _config()
    assert config.down_tile_m is None
    assert config.down_tile_n is None
    assert config.down_tile_k is None
    assert config.stage2_tile_m == config.tile_m
    assert config.stage2_tile_n == config.tile_n
    assert config.stage2_tile_k == config.tile_k
    assert config.route_tile_m == config.tile_m
    assert config.stage1_k_wave == 1
    split_m_config = replace(config, tile_m=32, down_tile_m=128)
    assert split_m_config.stage2_tile_m == 128
    assert split_m_config.route_tile_m == 128
    # Stage 2 overlays its A tile and FP32 epilogue scratch because their
    # lifetimes are disjoint.  Each 128 KiB region fits gfx950 even though
    # summing them would incorrectly reject this configuration as 256 KiB.
    overlay_config = SonicMoEConfig(
        hidden_size=512,
        intermediate_size=512,
        num_experts=4,
        top_k=2,
        tile_m=32,
        tile_n=128,
        tile_k=128,
        down_tile_m=128,
        down_tile_n=256,
        down_tile_k=512,
    )
    assert overlay_config.stage2_tile_m == 128
    with pytest.raises(ValueError, match="stage2 tile needs"):
        replace(overlay_config, hidden_size=1024, down_tile_n=512)
    candidate_m_tiles = {
        (candidate.tile_m, candidate.stage2_tile_m)
        for candidate in default_sonic_moe_candidates(config)
    }
    assert (32, 128) in candidate_m_tiles
    assert (64, 128) in candidate_m_tiles
    assert config.renormalize is True
    assert _config(num_experts=896, top_k=16).supports_flydsl_router
    assert config.stage2_output_mode == "atomic"
    assert replace(config, stage2_output_mode="reduce").stage2_output_mode == "reduce"

    invalid_configs = [
        {"hidden_size": 0},
        {"intermediate_size": 0},
        {"num_experts": 0},
        {"top_k": 0},
        {"top_k": NUM_EXPERTS + 1},
        {"tile_m": 0},
        {"down_tile_m": 0},
        {"tile_n": 0},
        {"tile_k": 0},
        {"hidden_size": HIDDEN_SIZE - 1},
        {"intermediate_size": INTERMEDIATE_SIZE - 1},
    ]
    for override in invalid_configs:
        with pytest.raises((TypeError, ValueError)):
            _config(**override)
    with pytest.raises(ValueError, match="unsupported activation"):
        _config(activation="not-an-activation")
    with pytest.raises(TypeError, match="activation must be a string"):
        _config(activation=None)
    with pytest.raises(ValueError, match="unsupported compute_dtype"):
        _config(compute_dtype="fp32")
    with pytest.raises(ValueError, match="unsupported stage2_output_mode"):
        _config(stage2_output_mode="not-a-mode")
    with pytest.raises(TypeError, match="stage2_output_mode must be a string"):
        _config(stage2_output_mode=None)

    atomic_workspace = SonicMoEWorkspace.allocate(config, TOKENS, device)
    reduce_config = replace(config, stage2_output_mode="reduce")
    reduce_workspace = SonicMoEWorkspace.allocate(reduce_config, TOKENS, device)
    ragged_reduce_workspace = SonicMoEWorkspace.allocate(
        reduce_config,
        TOKENS,
        device,
        routes=TOKENS * TOP_K,
    )
    assert atomic_workspace.route_output is None
    assert reduce_workspace.route_output is not None
    assert reduce_workspace.route_output.shape == (TOKENS, TOP_K, HIDDEN_SIZE)
    assert ragged_reduce_workspace.route_output is None

    large_mesh_config = _config(num_experts=300, top_k=1)
    with pytest.raises(ValueError, match="signed 32-bit byte-index"):
        SonicMoEWorkspace.allocate(large_mesh_config, 8_000_000, device)


def test_sonic_moe_stage2_launcher_cache_separates_output_modes(monkeypatch):
    config = _config(stage2_output_mode="reduce")
    compile_calls = []

    def fake_compile_gemm2(**kwargs):
        compile_calls.append(kwargs)
        return object()

    monkeypatch.setattr("kernels.moe.sonic.compile_gemm2_a16w4_port", fake_compile_gemm2)
    _get_stage2_launcher.cache_clear()
    try:
        atomic = _get_stage2_launcher(config, 0, "bf16", False, "atomic", 1, 0)
        assert _get_stage2_launcher(config, 0, "bf16", False, "atomic", 1, 0) is atomic
        reduce = _get_stage2_launcher(config, 0, "bf16", False, "reduce", 1, 0)
        assert _get_stage2_launcher(config, 0, "bf16", False, "reduce", 1, 0) is reduce

        assert atomic is not reduce
        assert [call["output_mode"] for call in compile_calls] == ["atomic", "reduce"]
        assert [call["TOPK"] for call in compile_calls] == [TOP_K, TOP_K]
        assert [call["BM"] for call in compile_calls] == [config.stage2_tile_m] * 2
        assert [call["SORTED_BM"] for call in compile_calls] == [config.route_tile_m] * 2
        assert _get_stage2_launcher.cache_info().currsize == 2
    finally:
        _get_stage2_launcher.cache_clear()


@pytest.mark.parametrize(
    "tuned",
    (
        SonicMoEConfig(
            hidden_size=4096,
            intermediate_size=2048,
            num_experts=64,
            top_k=8,
            tile_m=128,
            tile_n=256,
            tile_k=64,
            down_tile_m=128,
            down_tile_n=128,
            down_tile_k=64,
            stage2_xcd_swizzle=8,
        ),
        SonicMoEConfig(
            hidden_size=2048,
            intermediate_size=768,
            num_experts=128,
            top_k=8,
            tile_m=128,
            tile_n=192,
            tile_k=64,
            down_tile_m=64,
            down_tile_n=256,
            down_tile_k=64,
            stage1_xcd_swizzle=8,
            stage2_xcd_swizzle=0,
            stage1_write_padded_rows=True,
            stage1_lds_swizzle=True,
        ),
        SonicMoEConfig(
            hidden_size=4096,
            intermediate_size=14336,
            num_experts=8,
            top_k=2,
            tile_m=128,
            tile_n=256,
            tile_k=64,
            down_tile_m=128,
            down_tile_n=128,
            down_tile_k=64,
            stage1_xcd_swizzle=8,
            stage2_xcd_swizzle=8,
            stage1_write_padded_rows=True,
            stage1_lds_swizzle=True,
        ),
    ),
    ids=("e64-throughput", "e128-prefill", "e8-prefill"),
)
def test_sonic_moe_stage2_pipeline_gate_is_exact(tuned):
    assert _stage2_stages(tuned, 4096) == 2
    assert _stage2_stages(tuned, 4095) == 1
    assert _stage2_stages(tuned, 8192) == 1
    if tuned.num_experts in (8, 128):
        assert tuned.stage1_write_padded_rows
        assert tuned.stage1_lds_swizzle
        assert tuned.down_tile_k == 64
        assert _stage2_stages(replace(tuned, stage1_write_padded_rows=False), 4096) == 1
        assert _stage2_stages(replace(tuned, stage1_lds_swizzle=False), 4096) == 1
        assert _stage2_stages(replace(tuned, down_tile_k=128), 4096) == 1


def test_sonic_moe_stage2_pipeline_gate_rejects_other_tuning_axes():
    tuned = SonicMoEConfig(
        hidden_size=4096,
        intermediate_size=2048,
        num_experts=64,
        top_k=8,
        tile_m=128,
        tile_n=256,
        tile_k=64,
        down_tile_m=128,
        down_tile_n=128,
        down_tile_k=64,
        stage2_xcd_swizzle=8,
    )

    for fallback in (
        replace(tuned, hidden_size=3584),
        replace(tuned, intermediate_size=1024),
        replace(tuned, num_experts=128),
        replace(tuned, top_k=4),
        replace(tuned, down_tile_m=64),
        replace(tuned, down_tile_n=64),
        replace(tuned, down_tile_k=128),
        replace(tuned, tile_m=256),
        replace(tuned, stage2_xcd_swizzle=1),
        replace(tuned, stage2_b_cache_mod=2),
        replace(tuned, waves_per_eu=1),
        replace(tuned, persistent_stage2=True),
        replace(tuned, stage2_output_mode="reduce"),
        replace(tuned, compute_dtype="fp16"),
    ):
        assert _stage2_stages(fallback, 4096) == 1

    assert _stage2_stages(replace(tuned, stage2_pipeline_stages=1), 4096) == 1
    assert _stage2_stages(replace(tuned, stage2_pipeline_stages=2), 4095) == 2
    assert _stage2_stages(replace(tuned, stage2_pipeline_stages=2, stage2_output_mode="reduce"), 4096) == 1
    assert _stage2_stages(replace(tuned, stage2_pipeline_stages=2, compute_dtype="fp16"), 4096) == 1


@pytest.mark.parametrize("value", (0, 3))
def test_sonic_moe_rejects_invalid_stage2_pipeline_depth_value(value):
    with pytest.raises(ValueError, match="stage2_pipeline_stages"):
        _config(stage2_pipeline_stages=value)


@pytest.mark.parametrize("value", (True, 1.0, 2.0, "2"))
def test_sonic_moe_rejects_non_integer_stage2_pipeline_depth(value):
    with pytest.raises(TypeError, match="stage2_pipeline_stages"):
        _config(stage2_pipeline_stages=value)


@pytest.mark.parametrize("value", (None, 0, 1, "yes"))
def test_sonic_moe_rejects_non_boolean_stage1_padding_store(value):
    with pytest.raises(TypeError, match="stage1_write_padded_rows"):
        _config(stage1_write_padded_rows=value)


@pytest.mark.parametrize("value", (None, 0, 1, "yes"))
def test_sonic_moe_rejects_non_boolean_stage1_lds_swizzle(value):
    with pytest.raises(TypeError, match="stage1_lds_swizzle"):
        _config(stage1_lds_swizzle=value)


def test_sonic_moe_stage1_lds_swizzle_rejects_non_power_of_two_chunk_count():
    with pytest.raises(AssertionError, match="TILE_K/8 to be a power of two"):
        compile_gemm1_a16w4_port(
            BM=64,
            D_HIDDEN=192,
            D_INTER=128,
            NE=1,
            TOPK=1,
            TILE_N=128,
            TILE_K=96,
            w_dtype="bf16",
            a_dtype="bf16",
            a_lds_swizzle=True,
        )


def test_sonic_moe_stage2_pipeline_single_k_tile_normalizes_to_one():
    config = _config(stage2_pipeline_stages=2)
    assert config.intermediate_size == config.stage2_tile_k
    assert config.stage2_effective_pipeline_stages == 1
    assert _stage2_stages(config, 4096) == 1


def test_sonic_moe_stage2_pipeline_lds_is_validated_at_config_construction():
    serial = SonicMoEConfig(
        hidden_size=256,
        intermediate_size=512,
        num_experts=4,
        top_k=2,
        tile_m=16,
        tile_n=128,
        tile_k=128,
        down_tile_m=256,
        down_tile_n=128,
        down_tile_k=256,
        stage2_pipeline_stages=1,
    )
    assert serial.stage2_effective_pipeline_stages == 1
    with pytest.raises(ValueError, match="stage2 tile needs"):
        replace(serial, stage2_pipeline_stages=2)


def test_sonic_moe_stage2_pipeline_field_preserves_legacy_positional_abi():
    config = SonicMoEConfig(
        256,
        128,
        4,
        2,
        16,
        128,
        128,
        None,
        None,
        None,
        False,
        0,
        2,
        3,
        1,
        4,
        None,
        False,
        "reduce",
        "relu",
        "fp16",
    )
    assert config.stage2_output_mode == "reduce"
    assert config.activation == "relu"
    assert config.compute_dtype == "fp16"
    assert config.stage2_pipeline_stages is None


def test_sonic_moe_stage2_pipeline_gate_rejects_ragged_routes(monkeypatch):
    """Flat route lists must retain the measured serial Stage-2 path."""

    import kernels.moe.sonic as sonic_module

    config = _config()
    x, w1, w2, router_logits = _make_case(seed=269)
    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    selected_stages = []
    get_stage2_launcher = sonic_module._get_stage2_launcher

    monkeypatch.setattr(sonic_module, "_stage2_stages", lambda _config, _tokens: 2)

    def tracking_get_stage2_launcher(config, b_cache_mod, weight_dtype, has_bias, output_mode, stages, device_index):
        selected_stages.append(stages)
        return get_stage2_launcher(
            config,
            b_cache_mod,
            weight_dtype,
            has_bias,
            output_mode,
            stages,
            device_index,
        )

    monkeypatch.setattr(sonic_module, "_get_stage2_launcher", tracking_get_stage2_launcher)
    fixed = op.forward_topk(x, topk_ids, topk_weights)
    ragged = op.forward_routes(
        x,
        torch.arange(TOKENS, dtype=torch.int32, device=x.device).repeat_interleave(TOP_K),
        topk_ids.reshape(-1).contiguous(),
        topk_weights.reshape(-1).contiguous(),
    )
    torch.cuda.synchronize()

    assert selected_stages == [2, 1]
    _assert_close(fixed, ragged)


def test_sonic_moe_stage2_launcher_cache_separates_pipeline_depth(monkeypatch):
    config = _config()
    compile_calls = []

    def fake_compile_gemm2(**kwargs):
        compile_calls.append(kwargs)
        return object()

    monkeypatch.setattr("kernels.moe.sonic.compile_gemm2_a16w4_port", fake_compile_gemm2)
    _get_stage2_launcher.cache_clear()
    try:
        serial = _get_stage2_launcher(config, 0, "bf16", False, "atomic", 1, 0)
        pipeline = _get_stage2_launcher(config, 0, "bf16", False, "atomic", 2, 0)

        assert serial is not pipeline
        assert [call["stages"] for call in compile_calls] == [1, 2]
        assert _get_stage2_launcher.cache_info().currsize == 2
    finally:
        _get_stage2_launcher.cache_clear()


def test_sonic_moe_launchers_receive_independent_compute_and_route_tiles(monkeypatch):
    config = _config(tile_m=32, down_tile_m=128, stage1_k_wave=2)
    compile_calls = {}

    def fake_compile_gemm1(**kwargs):
        compile_calls["stage1"] = kwargs
        return object()

    def fake_compile_gemm2(**kwargs):
        compile_calls["stage2"] = kwargs
        return object()

    monkeypatch.setattr("kernels.moe.sonic.compile_gemm1_a16w4_port", fake_compile_gemm1)
    monkeypatch.setattr("kernels.moe.sonic.compile_gemm2_a16w4_port", fake_compile_gemm2)
    _get_stage1_launcher.cache_clear()
    _get_stage2_launcher.cache_clear()
    try:
        _get_stage1_launcher(config, 0, "bf16", False, 0)
        _get_stage2_launcher(config, 0, "bf16", False, "atomic", 1, 0)
        assert compile_calls["stage1"]["BM"] == 32
        assert compile_calls["stage1"]["SORTED_BM"] == 128
        assert compile_calls["stage1"]["k_wave"] == 2
        assert compile_calls["stage2"]["BM"] == 128
        assert compile_calls["stage2"]["SORTED_BM"] == 128
    finally:
        _get_stage1_launcher.cache_clear()
        _get_stage2_launcher.cache_clear()


def test_sonic_moe_stage1_launcher_cache_separates_k_wave(monkeypatch):
    compile_calls = []

    def fake_compile_gemm1(**kwargs):
        compile_calls.append(kwargs)
        return object()

    monkeypatch.setattr("kernels.moe.sonic.compile_gemm1_a16w4_port", fake_compile_gemm1)
    _get_stage1_launcher.cache_clear()
    try:
        default_config = _config()
        split_k_config = replace(default_config, stage1_k_wave=2)
        default_launcher = _get_stage1_launcher(default_config, 0, "bf16", False, 0)
        assert _get_stage1_launcher(default_config, 0, "bf16", False, 0) is default_launcher
        split_k_launcher = _get_stage1_launcher(split_k_config, 0, "bf16", False, 0)
        assert _get_stage1_launcher(split_k_config, 0, "bf16", False, 0) is split_k_launcher

        assert default_launcher is not split_k_launcher
        assert [call["k_wave"] for call in compile_calls] == [1, 2]
        assert _get_stage1_launcher.cache_info().currsize == 2
    finally:
        _get_stage1_launcher.cache_clear()


def test_sonic_moe_tensor_shape_and_dtype_validation():
    config = _config()
    x, w1, w2, router_logits = _make_case()

    with pytest.raises((TypeError, ValueError)):
        prepare_sonic_bf16_weights(w1[:, :-1, :], w2, config)
    with pytest.raises((TypeError, ValueError)):
        prepare_sonic_bf16_weights(w1, w2[:, :, :-1], config)
    prepared_fp32 = prepare_sonic_bf16_weights(w1.float(), w2.float(), config)
    assert prepared_fp32.gate_up.dtype == torch.bfloat16
    assert prepared_fp32.down.dtype == torch.bfloat16
    with pytest.raises((TypeError, ValueError)):
        prepare_sonic_bf16_weights(w1.to(torch.int16), w2, config)

    b1 = torch.randn(
        (NUM_EXPERTS, config.stage1_projection_size),
        dtype=torch.bfloat16,
        device=x.device,
    )
    b2 = torch.randn((NUM_EXPERTS, HIDDEN_SIZE), dtype=torch.bfloat16, device=x.device)
    with pytest.raises(ValueError, match="both be provided"):
        prepare_sonic_bf16_weights(w1, w2, config, b1=b1)
    with pytest.raises(ValueError, match="b1 must have shape"):
        prepare_sonic_bf16_weights(w1, w2, config, b1=b1[:, :-1], b2=b2)
    with pytest.raises(ValueError, match="b2 must have shape"):
        prepare_sonic_bf16_weights(w1, w2, config, b1=b1, b2=b2[:, :-1])
    with pytest.raises(TypeError, match="b1/b2 must be floating point"):
        prepare_sonic_bf16_weights(w1, w2, config, b1=b1.to(torch.int16), b2=b2)

    prepared_bias = prepare_sonic_bf16_weights(w1, w2, config, b1=b1, b2=b2)
    prepared_b1 = prepared_bias.stage1_bias.clone()
    prepared_b2 = prepared_bias.stage2_bias.clone()
    b1.zero_()
    b2.zero_()
    assert torch.equal(prepared_bias.stage1_bias, prepared_b1)
    assert torch.equal(prepared_bias.stage2_bias, prepared_b2)

    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    with pytest.raises((TypeError, ValueError)):
        op(x[:, :-1], router_logits)
    with pytest.raises((TypeError, ValueError)):
        op(x.float(), router_logits)
    misaligned_x_storage = torch.empty(TOKENS * HIDDEN_SIZE + 1, device=x.device, dtype=torch.bfloat16)
    misaligned_x = misaligned_x_storage[1:].view(TOKENS, HIDDEN_SIZE)
    assert misaligned_x.is_contiguous() and misaligned_x.data_ptr() % 16
    with pytest.raises(ValueError, match="16-byte aligned"):
        op(misaligned_x, router_logits)
    with pytest.raises((TypeError, ValueError)):
        op(x, router_logits[:, :-1])
    with pytest.raises((TypeError, ValueError)):
        op(x, router_logits[:-1])
    with pytest.raises((TypeError, ValueError)):
        op(
            x,
            router_logits,
            out=torch.empty((TOKENS, HIDDEN_SIZE - 1), device=x.device, dtype=x.dtype),
        )
    with pytest.raises((TypeError, ValueError)):
        op(x, router_logits, out=x)
    misaligned_out_storage = torch.empty(TOKENS * HIDDEN_SIZE + 1, device=x.device, dtype=torch.bfloat16)
    misaligned_out = misaligned_out_storage[1:].view(TOKENS, HIDDEN_SIZE)
    assert misaligned_out.is_contiguous() and misaligned_out.data_ptr() % 4 == 2
    with pytest.raises(ValueError, match="4-byte aligned"):
        op(x, router_logits, out=misaligned_out)
    with pytest.raises((TypeError, ValueError)):
        op(
            x,
            router_logits,
            out=torch.empty_like(x, requires_grad=True),
        )
    workspace = op.reserve(TOKENS)
    internal_out_alias = workspace.intermediate.flatten()[: TOKENS * HIDDEN_SIZE].view(TOKENS, HIDDEN_SIZE)
    with pytest.raises((TypeError, ValueError)):
        op(x, router_logits, out=internal_out_alias)
    with pytest.raises((TypeError, ValueError)):
        op(x.detach().requires_grad_(True), router_logits)
    with pytest.raises((TypeError, ValueError)):
        op(x, router_logits.detach().requires_grad_(True))

    topk_ids, topk_weights = _topk_from_logits(router_logits, config)
    with pytest.raises((TypeError, ValueError)):
        op.forward_topk(x, topk_ids[:, :1], topk_weights)
    with pytest.raises((TypeError, ValueError)):
        op.forward_topk(x, topk_ids, topk_weights[:, :1])


def test_sonic_moe_fp16_preparation_and_dtype_validation():
    bf16_config = _config()
    fp16_config = _config(compute_dtype="fp16")
    x, w1, w2, router_logits = _make_case(dtype=torch.float16)
    b1 = torch.randn(
        (NUM_EXPERTS, fp16_config.stage1_projection_size),
        dtype=torch.float16,
        device=x.device,
    )
    b2 = torch.randn((NUM_EXPERTS, HIDDEN_SIZE), dtype=torch.float16, device=x.device)

    prepared = prepare_sonic_fp16_weights(w1.float(), w2.float(), fp16_config, b1=b1, b2=b2)
    assert prepared.weight_dtype == "fp16"
    assert prepared.compute_dtype == torch.float16
    assert prepared.gate_up.dtype == torch.float16
    assert prepared.down.dtype == torch.float16
    assert prepared.stage1_bias is not None and prepared.stage1_bias.dtype == torch.float16
    assert prepared.stage2_bias is not None and prepared.stage2_bias.dtype == torch.float16

    op = SonicMoE(fp16_config, prepared)
    workspace = op.reserve(TOKENS)
    assert workspace.intermediate.dtype == torch.float16
    assert workspace.output.dtype == torch.float16
    with pytest.raises(TypeError, match="hidden_states must use"):
        op(x.to(torch.bfloat16), router_logits)
    with pytest.raises(ValueError, match="compute_dtype='bf16'"):
        prepare_sonic_bf16_weights(w1, w2, fp16_config)
    with pytest.raises(ValueError, match="compute_dtype='fp16'"):
        prepare_sonic_fp16_weights(w1, w2, bf16_config)
    with pytest.raises(ValueError, match="require config.compute_dtype='bf16'"):
        prepare_sonic_mxfp4_weights(w1, w2, fp16_config)


def test_sonic_moe_rejects_malformed_prepared_storage_and_mxfp4_tiles():
    config = _config()
    _, w1, w2, _ = _make_case()
    prepared = prepare_sonic_bf16_weights(w1, w2, config)
    with pytest.raises(ValueError, match="prepared gate/up storage"):
        SonicMoE(config, replace(prepared, gate_up=prepared.gate_up.flatten()[:16]))
    with pytest.raises(TypeError, match="dummy_scale"):
        SonicMoE(
            config,
            replace(prepared, dummy_scale=prepared.dummy_scale.to(torch.int32)),
        )

    mxfp4 = prepare_sonic_mxfp4_weights(w1, w2, config)
    assert mxfp4.gate_up_scale is not None
    with pytest.raises(ValueError, match="wrong padded size"):
        SonicMoE(
            config,
            replace(mxfp4, gate_up_scale=mxfp4.gate_up_scale[:-4]),
        )
    nonfinite_w1 = w1.clone()
    nonfinite_w1[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        prepare_sonic_mxfp4_weights(nonfinite_w1, w2, config)

    tile64 = _config(tile_m=32, tile_k=64, down_tile_k=64)
    tile64_weights = prepare_sonic_mxfp4_weights(w1, w2, tile64)
    with pytest.raises(ValueError, match=">= 128"):
        SonicMoE(tile64, tile64_weights)

    format_specific_limit = SonicMoEConfig(
        hidden_size=32768,
        intermediate_size=32768,
        num_experts=1,
        top_k=1,
        tile_m=16,
        tile_n=128,
        tile_k=128,
    )
    tiny_bf16 = torch.empty(16, dtype=torch.bfloat16, device=w1.device)
    unsafe_bf16 = SonicMoEWeights(
        gate_up=tiny_bf16,
        down=tiny_bf16,
        dummy_scale=torch.zeros(1, dtype=torch.uint8, device=w1.device),
        config=format_specific_limit,
    )
    with pytest.raises(ValueError, match="dense 16-bit gate/up weights"):
        SonicMoE(format_specific_limit, unsafe_bf16)

    huge_expert_config = _config(num_experts=2_097_153)
    tiny = torch.empty(16, dtype=torch.uint8, device=w1.device)
    unsafe = SonicMoEWeights(
        gate_up=tiny,
        down=tiny,
        dummy_scale=tiny[:1],
        config=huge_expert_config,
        gate_up_scale=tiny,
        down_scale=tiny,
        weight_dtype="mxfp4",
    )
    with pytest.raises(ValueError, match="32-bit buffer-offset limit"):
        SonicMoE(huge_expert_config, unsafe)
