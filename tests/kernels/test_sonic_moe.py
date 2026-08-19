# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Correctness and API-contract tests for gfx950 SonicMoE A16W16/A16W4."""

import math
from dataclasses import replace

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.moe.sonic import (
    SonicMoE,
    SonicMoEConfig,
    SonicMoEWeights,
    SonicMoEWorkspace,
    _quantize_mxfp4_weight,
    prepare_sonic_bf16_weights,
    prepare_sonic_mxfp4_weights,
    sonic_moe_mxfp4_reference,
    sonic_moe_reference,
)
from kernels.moe.sonic_autotune import SonicMoEAutotuner

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


def _make_case(tokens=TOKENS, seed=17):
    device = _gfx950_device()
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        (tokens, HIDDEN_SIZE), device=device, dtype=torch.float32, generator=generator
    ).to(torch.bfloat16)
    w1 = (
        torch.randn(
            (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        / math.sqrt(HIDDEN_SIZE)
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(
            (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        / math.sqrt(INTERMEDIATE_SIZE)
    ).to(torch.bfloat16)
    router_logits = torch.randn(
        (tokens, NUM_EXPERTS), device=device, dtype=torch.float32, generator=generator
    ).to(torch.bfloat16)
    return x, w1, w2, router_logits


def _topk_from_logits(router_logits, config):
    scores = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(scores, config.top_k, dim=-1)
    if config.renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_ids.to(torch.int32), topk_weights.contiguous()


def _assert_close(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert actual.device == expected.device
    torch.testing.assert_close(actual.float(), expected.float(), rtol=3e-2, atol=5e-2)


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
    w1 = (
        torch.randn((4, 512, 256), device=device, dtype=torch.float32, generator=generator)
        / math.sqrt(256)
    ).to(torch.bfloat16)
    w2 = (
        torch.randn((4, 256, 256), device=device, dtype=torch.float32, generator=generator)
        / math.sqrt(256)
    ).to(torch.bfloat16)
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
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    ).item()
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
    reference_scale = gemm_common_utils.f32_to_e8m0(
        blocks.abs().amax(dim=1) / 4.0
    ).view(torch.uint8)
    reference_values = blocks / gemm_common_utils.e8m0_to_f32(reference_scale)[:, None]
    reference_packed = (
        gemm_common_utils.f32_to_mxfp4(reference_values)
        .view(torch.uint8)
        .view_as(packed)
    )
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

    config = _config(num_experts=896, top_k=2, tile_m=32)
    generator = torch.Generator(device=device).manual_seed(53)
    x = torch.randn((1, HIDDEN_SIZE), dtype=torch.bfloat16, device=device, generator=generator)
    w1 = (
        torch.randn(
            (896, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / math.sqrt(HIDDEN_SIZE)
    )
    w2 = (
        torch.randn(
            (896, HIDDEN_SIZE, INTERMEDIATE_SIZE),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / math.sqrt(INTERMEDIATE_SIZE)
    )
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
    assert '"version": 2' in non_object_cache_file.read_text(encoding="utf-8")


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


def test_sonic_moe_uses_independent_workspaces_across_streams():
    config = _config()
    x_a, w1, w2, logits_a = _make_case(seed=37)
    x_b, _, _, logits_b = _make_case(seed=41)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    expected_a = sonic_moe_reference(x_a, w1, w2, logits_a, config)
    expected_b = sonic_moe_reference(x_b, w1, w2, logits_b, config)

    default_stream = torch.cuda.current_stream(x_a.device)
    stream_a = torch.cuda.Stream(device=x_a.device)
    stream_b = torch.cuda.Stream(device=x_a.device)
    stream_a.wait_stream(default_stream)
    stream_b.wait_stream(default_stream)

    with torch.cuda.stream(stream_a):
        actual_a = op(x_a, logits_a)
        workspace_a = op.workspace
    with torch.cuda.stream(stream_b):
        actual_b = op(x_b, logits_b)
        workspace_b = op.workspace

    torch.cuda.synchronize()
    assert workspace_a is not None
    assert workspace_b is not None
    assert workspace_a is not workspace_b
    _assert_close(actual_a, expected_a)
    _assert_close(actual_b, expected_b)


def test_sonic_moe_non_power_of_two_router_fallback():
    """Arbitrary expert counts use torch top-k but keep FlyDSL sort/GEMMs."""

    device = _gfx950_device()
    config = _config(num_experts=6)
    assert not config.supports_flydsl_router
    generator = torch.Generator(device=device).manual_seed(29)
    x = torch.randn((3, HIDDEN_SIZE), device=device, dtype=torch.bfloat16, generator=generator)
    w1 = torch.randn(
        (6, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ) / math.sqrt(HIDDEN_SIZE)
    w2 = torch.randn(
        (6, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ) / math.sqrt(INTERMEDIATE_SIZE)
    logits = torch.randn((3, 6), device=device, dtype=torch.bfloat16, generator=generator)
    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))

    expected = sonic_moe_reference(x, w1, w2, logits, config)
    actual = op(x, logits)
    torch.cuda.synchronize()
    _assert_close(actual, expected)


def test_sonic_moe_config_validation():
    device = _gfx950_device()
    config = _config()
    assert config.down_tile_n is None
    assert config.down_tile_k is None
    assert config.stage2_tile_n == config.tile_n
    assert config.stage2_tile_k == config.tile_k
    assert config.renormalize is True

    invalid_configs = [
        {"hidden_size": 0},
        {"intermediate_size": 0},
        {"num_experts": 0},
        {"top_k": 0},
        {"top_k": NUM_EXPERTS + 1},
        {"tile_m": 0},
        {"tile_n": 0},
        {"tile_k": 0},
        {"hidden_size": HIDDEN_SIZE - 1},
        {"intermediate_size": INTERMEDIATE_SIZE - 1},
    ]
    for override in invalid_configs:
        with pytest.raises((TypeError, ValueError)):
            _config(**override)

    large_mesh_config = _config(num_experts=300, top_k=1)
    with pytest.raises(ValueError, match="signed 32-bit byte-index"):
        SonicMoEWorkspace.allocate(large_mesh_config, 8_000_000, device)


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

    op = SonicMoE(config, prepare_sonic_bf16_weights(w1, w2, config))
    with pytest.raises((TypeError, ValueError)):
        op(x[:, :-1], router_logits)
    with pytest.raises((TypeError, ValueError)):
        op(x.float(), router_logits)
    misaligned_x_storage = torch.empty(
        TOKENS * HIDDEN_SIZE + 1, device=x.device, dtype=torch.bfloat16
    )
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
    misaligned_out_storage = torch.empty(
        TOKENS * HIDDEN_SIZE + 1, device=x.device, dtype=torch.bfloat16
    )
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
    internal_out_alias = workspace.intermediate.flatten()[: TOKENS * HIDDEN_SIZE].view(
        TOKENS, HIDDEN_SIZE
    )
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
    with pytest.raises(ValueError, match="BF16 gate/up weights"):
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
