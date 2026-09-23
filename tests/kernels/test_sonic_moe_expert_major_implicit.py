# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for E16 expert-major routing without materialized route IDs."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.moe import moe_ragged_sorting_kernel as route_sorting_module
from kernels.moe import sonic_backward as sonic_backward_module
from kernels.moe.sonic import (
    SonicMoE,
    SonicMoEConfig,
    prepare_sonic_bf16_weights,
    sonic_moe_backward_expert_major,
    sonic_moe_backward_routes,
)


pytestmark = [
    pytest.mark.l2_device,
    pytest.mark.rocm_lower,
    pytest.mark.large_shape,
]

_DYNAMIC_ROUTES = (0, 1, 63, 64, 127, 128, 257, 4097, 8191, 8192, 9000)
_STATE_VALUE_FIELDS = (
    "tokens",
    "routes",
    "hidden_size",
    "intermediate_size",
    "num_experts",
    "activation",
    "compute_dtype",
    "interleaved_w1",
    "has_bias",
    "expert_major",
    "token_indices_identity",
    "route_policy_size",
)


def _gfx950_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU is required")
    arch = str(get_rocm_arch())
    if "gfx950" not in arch:
        pytest.skip(
            "implicit E16 expert-major tests require gfx950, "
            f"found {arch}"
        )
    return torch.device("cuda")


def _qwen3_e16_config() -> SonicMoEConfig:
    return SonicMoEConfig(
        hidden_size=2048,
        intermediate_size=768,
        num_experts=16,
        top_k=1,
        tile_m=128,
        tile_n=192,
        tile_k=64,
        down_tile_m=64,
        down_tile_n=256,
        down_tile_k=64,
        stage1_xcd_swizzle=8,
        stage2_xcd_swizzle=0,
        stage2_pipeline_stages=2,
        stage1_write_padded_rows=True,
        stage1_lds_swizzle=True,
        renormalize=False,
    )


@pytest.fixture(scope="module")
def e16_operators():
    device = _gfx950_device()
    config = _qwen3_e16_config()
    generator = torch.Generator(device=device).manual_seed(20260922)
    w1 = torch.randn(
        (
            config.num_experts,
            2 * config.intermediate_size,
            config.hidden_size,
        ),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.02)
    w2 = torch.randn(
        (
            config.num_experts,
            config.hidden_size,
            config.intermediate_size,
        ),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.02)
    prepared = prepare_sonic_bf16_weights(w1, w2, config)
    return SimpleNamespace(
        device=device,
        config=config,
        w1=w1,
        w2=w2,
        materialized=SonicMoE(config, prepared),
        implicit=SonicMoE(config, prepared),
        counts=SonicMoE(config, prepared),
    )


def _make_expert_major_case(case, routes: int):
    generator = torch.Generator(device=case.device).manual_seed(1000 + routes)
    hidden = torch.randn(
        (routes, case.config.hidden_size),
        dtype=torch.bfloat16,
        device=case.device,
        generator=generator,
    ).mul_(0.2)
    grad_output = torch.randn(
        (routes, case.config.hidden_size),
        dtype=torch.bfloat16,
        device=case.device,
        generator=generator,
    ).mul_(0.2)

    rows_per_expert, remainder = divmod(routes, case.config.num_experts)
    counts = tuple(
        rows_per_expert + int(expert < remainder)
        for expert in range(case.config.num_experts)
    )
    offsets_host = [0]
    for count in counts:
        offsets_host.append(offsets_host[-1] + count)
    expert_offsets = torch.tensor(
        offsets_host,
        dtype=torch.int32,
        device=case.device,
    )
    expert_counts = torch.tensor(
        counts,
        dtype=torch.int32,
        device=case.device,
    )
    token_indices = torch.arange(
        routes,
        dtype=torch.int32,
        device=case.device,
    )
    expert_indices = torch.repeat_interleave(
        torch.arange(
            case.config.num_experts,
            dtype=torch.int32,
            device=case.device,
        ),
        torch.tensor(counts, dtype=torch.int64, device=case.device),
    ).contiguous()
    route_weights = torch.linspace(
        0.25,
        1.0,
        routes,
        dtype=torch.float32,
        device=case.device,
    )
    return SimpleNamespace(
        hidden=hidden,
        grad_output=grad_output,
        counts=counts,
        expert_counts=expert_counts,
        expert_offsets=expert_offsets,
        token_indices=token_indices,
        expert_indices=expert_indices,
        route_weights=route_weights,
    )


def _assert_states_equal(materialized, implicit, *, route_tile_m: int) -> None:
    for field in _STATE_VALUE_FIELDS:
        assert getattr(implicit, field) == getattr(materialized, field), field
    assert implicit.ready_event.query()
    assert materialized.ready_event.query()
    for field in (
        "preactivation",
        "num_valid_ids",
        "expert_frequency",
    ):
        implicit_tensor = getattr(implicit, field)
        materialized_tensor = getattr(materialized, field)
        assert isinstance(implicit_tensor, torch.Tensor), field
        assert isinstance(materialized_tensor, torch.Tensor), field
        assert torch.equal(implicit_tensor, materialized_tensor), field

    # Retained route-metadata arrays are capacity-sized.  Only the prefix
    # identified by num_valid_ids is initialized; comparing the unused tail
    # would compare allocator garbage rather than state semantics.
    padded_rows = int(implicit.num_valid_ids[0].item())
    padded_blocks = padded_rows // route_tile_m
    for field in ("sorted_token_ids", "sorted_route_ids", "sorted_weights"):
        implicit_tensor = getattr(implicit, field)
        materialized_tensor = getattr(materialized, field)
        assert isinstance(implicit_tensor, torch.Tensor), field
        assert isinstance(materialized_tensor, torch.Tensor), field
        assert tuple(implicit_tensor.shape) == tuple(materialized_tensor.shape)
        assert torch.equal(
            implicit_tensor[:padded_rows],
            materialized_tensor[:padded_rows],
        ), field
    assert isinstance(implicit.sorted_expert_ids, torch.Tensor)
    assert isinstance(materialized.sorted_expert_ids, torch.Tensor)
    assert tuple(implicit.sorted_expert_ids.shape) == tuple(
        materialized.sorted_expert_ids.shape
    )
    assert torch.equal(
        implicit.sorted_expert_ids[:padded_blocks],
        materialized.sorted_expert_ids[:padded_blocks],
    )


@pytest.mark.parametrize("routes", _DYNAMIC_ROUTES)
def test_expert_major_training_without_ids_matches_materialized_path(
    e16_operators,
    monkeypatch,
    routes,
):
    """Every dynamic bucket preserves output, retained state, and gradients."""

    values = _make_expert_major_case(e16_operators, routes)
    expected_frequency = torch.tensor(
        values.counts,
        dtype=torch.int32,
        device=e16_operators.device,
    )
    materialized_frequency = torch.empty_like(expected_frequency)
    implicit_frequency = torch.empty_like(expected_frequency)
    counts_frequency = torch.empty_like(expected_frequency)

    materialized_output, materialized_state = (
        e16_operators.materialized.forward_routes_training(
            values.hidden,
            values.token_indices,
            values.expert_indices,
            values.route_weights,
            expert_offsets=values.expert_offsets,
            token_indices_identity=True,
            expert_frequency_out=materialized_frequency,
        )
    )
    implicit_output, implicit_state = (
        e16_operators.implicit.forward_expert_major_training(
            values.hidden,
            values.expert_offsets,
            values.route_weights,
            expert_frequency_out=implicit_frequency,
        )
    )
    counts_output, counts_state = (
        e16_operators.counts.forward_expert_major_counts_training(
            values.hidden,
            values.expert_counts,
            values.route_weights,
            expert_frequency_out=counts_frequency,
        )
    )
    torch.cuda.synchronize(e16_operators.device)

    assert torch.equal(implicit_output, materialized_output)
    assert torch.equal(counts_output, implicit_output)
    assert torch.equal(materialized_frequency, expected_frequency)
    assert torch.equal(implicit_frequency, expected_frequency)
    assert torch.equal(counts_frequency, expected_frequency)
    _assert_states_equal(
        materialized_state,
        implicit_state,
        route_tile_m=e16_operators.config.route_tile_m,
    )
    _assert_states_equal(
        implicit_state,
        counts_state,
        route_tile_m=e16_operators.config.route_tile_m,
    )

    def _unexpected_sort(*_args, **_kwargs):
        raise AssertionError("complete retained identity state must not re-sort")

    monkeypatch.setattr(
        sonic_backward_module,
        "moe_ragged_sorting_flydsl",
        _unexpected_sort,
    )
    materialized_gradients = sonic_moe_backward_routes(
        values.hidden,
        e16_operators.w1,
        e16_operators.w2,
        values.token_indices,
        values.expert_indices,
        values.route_weights,
        values.grad_output,
        e16_operators.config,
        forward_state=materialized_state,
        token_indices_identity=True,
    )
    implicit_gradients = sonic_moe_backward_expert_major(
        values.hidden,
        e16_operators.w1,
        e16_operators.w2,
        values.route_weights,
        values.grad_output,
        e16_operators.config,
        forward_state=implicit_state,
    )
    counts_gradients = sonic_moe_backward_expert_major(
        values.hidden,
        e16_operators.w1,
        e16_operators.w2,
        values.route_weights,
        values.grad_output,
        e16_operators.config,
        forward_state=counts_state,
    )
    torch.cuda.synchronize(e16_operators.device)

    for implicit_gradient, counts_gradient, materialized_gradient in zip(
        implicit_gradients,
        counts_gradients,
        materialized_gradients,
    ):
        assert torch.equal(implicit_gradient, materialized_gradient)
        assert torch.equal(counts_gradient, implicit_gradient)


def test_expert_major_inference_without_ids_matches_materialized_path(
    e16_operators,
):
    values = _make_expert_major_case(e16_operators, 257)
    materialized_frequency = torch.empty(
        e16_operators.config.num_experts,
        dtype=torch.int32,
        device=e16_operators.device,
    )
    implicit_frequency = torch.empty_like(materialized_frequency)
    counts_frequency = torch.empty_like(materialized_frequency)

    materialized_output = e16_operators.materialized.forward_routes(
        values.hidden,
        values.token_indices,
        values.expert_indices,
        values.route_weights,
        expert_offsets=values.expert_offsets,
        token_indices_identity=True,
        expert_frequency_out=materialized_frequency,
    )
    implicit_output = e16_operators.implicit.forward_expert_major(
        values.hidden,
        values.expert_offsets,
        values.route_weights,
        expert_frequency_out=implicit_frequency,
    )
    counts_output = e16_operators.counts.forward_expert_major_counts(
        values.hidden,
        values.expert_counts,
        values.route_weights,
        expert_frequency_out=counts_frequency,
    )
    torch.cuda.synchronize(e16_operators.device)

    assert torch.equal(implicit_output, materialized_output)
    assert torch.equal(counts_output, implicit_output)
    assert torch.equal(implicit_frequency, materialized_frequency)
    assert torch.equal(counts_frequency, implicit_frequency)


@pytest.mark.parametrize(
    "entrypoint",
    ("forward_expert_major_counts", "forward_expert_major_counts_training"),
)
@pytest.mark.parametrize(
    ("invalid_counts", "error_match"),
    (
        ("shape", "expert_counts must have shape"),
        ("dtype", "expert_counts must be contiguous int32"),
        ("device", "expert_counts must be contiguous int32"),
    ),
)
def test_expert_major_counts_validates_metadata_tensor(
    e16_operators,
    entrypoint,
    invalid_counts,
    error_match,
):
    values = _make_expert_major_case(e16_operators, 1)
    if invalid_counts == "shape":
        expert_counts = values.expert_counts[:-1]
    elif invalid_counts == "dtype":
        expert_counts = values.expert_counts.to(torch.int64)
    else:
        expert_counts = values.expert_counts.cpu()

    with pytest.raises(ValueError, match=error_match):
        getattr(e16_operators.counts, entrypoint)(
            values.hidden,
            expert_counts,
            values.route_weights,
        )


@pytest.mark.parametrize("training", (False, True), ids=("inference", "training"))
def test_expert_offsets_and_counts_are_mutually_exclusive(
    e16_operators,
    training,
):
    values = _make_expert_major_case(e16_operators, 1)
    common = dict(
        expert_offsets=values.expert_offsets,
        expert_counts=values.expert_counts,
        token_indices_identity=True,
        route_policy_size=None,
    )

    with pytest.raises(
        ValueError,
        match="expert_offsets and expert_counts are mutually exclusive",
    ):
        if training:
            e16_operators.counts._forward_routes_training_on_current_device(
                values.hidden,
                None,
                None,
                values.route_weights,
                None,
                interleaved_w1=False,
                expert_frequency_out=None,
                **common,
            )
        else:
            e16_operators.counts._forward_routes_on_current_device(
                values.hidden,
                None,
                None,
                values.route_weights,
                None,
                None,
                **common,
            )


def test_expert_major_backward_rejects_incomplete_or_damaged_state(
    e16_operators,
    monkeypatch,
):
    """The ID-free API must never silently fall back to a fresh route sort."""

    values = _make_expert_major_case(e16_operators, 257)
    _, state = e16_operators.implicit.forward_expert_major_training(
        values.hidden,
        values.expert_offsets,
        values.route_weights,
    )
    torch.cuda.synchronize(e16_operators.device)
    state_values = dict(vars(state))

    missing_metadata = dict(state_values)
    missing_metadata.pop("sorted_route_ids")
    wrong_metadata_shape = dict(state_values)
    wrong_metadata_shape["sorted_weights"] = state.sorted_weights[:-1]
    not_expert_major = dict(state_values)
    not_expert_major["expert_major"] = False
    not_identity = dict(state_values)
    not_identity["token_indices_identity"] = False
    damaged_states = (
        SimpleNamespace(**missing_metadata),
        SimpleNamespace(**wrong_metadata_shape),
        SimpleNamespace(**not_expert_major),
        SimpleNamespace(**not_identity),
    )

    def _unexpected_sort(*_args, **_kwargs):
        raise AssertionError("ID-free backward must fail before any sorter launch")

    monkeypatch.setattr(
        sonic_backward_module,
        "moe_ragged_sorting_flydsl",
        _unexpected_sort,
    )
    for damaged_state in damaged_states:
        with pytest.raises(
            ValueError,
            match="implicit expert-major backward requires a complete retained state",
        ):
            sonic_moe_backward_expert_major(
                values.hidden,
                e16_operators.w1,
                e16_operators.w2,
                values.route_weights,
                values.grad_output,
                e16_operators.config,
                forward_state=damaged_state,
            )


def test_expert_major_without_ids_is_independent_of_legacy_sorter_gate(
    e16_operators,
    monkeypatch,
):
    """No-ID APIs use fused Stage 1 even if the old sorter path is disabled."""

    values = _make_expert_major_case(e16_operators, 1)
    monkeypatch.setattr(
        route_sorting_module,
        "_E16_EXPERT_MAJOR_SINGLE_LAUNCH",
        False,
    )

    offsets_output = e16_operators.implicit.forward_expert_major(
        values.hidden,
        values.expert_offsets,
        values.route_weights,
    )
    offsets_training_output, _ = (
        e16_operators.implicit.forward_expert_major_training(
            values.hidden,
            values.expert_offsets,
            values.route_weights,
        )
    )
    counts_output = e16_operators.counts.forward_expert_major_counts(
        values.hidden,
        values.expert_counts,
        values.route_weights,
    )
    counts_training_output, _ = (
        e16_operators.counts.forward_expert_major_counts_training(
            values.hidden,
            values.expert_counts,
            values.route_weights,
        )
    )
    torch.cuda.synchronize(values.hidden.device)

    assert torch.equal(offsets_training_output, offsets_output)
    assert torch.equal(counts_output, offsets_output)
    assert torch.equal(counts_training_output, offsets_output)


@pytest.mark.parametrize(
    "entrypoint",
    ("forward_expert_major_training", "forward_expert_major_counts_training"),
)
def test_expert_major_training_rejects_unsupported_backward_lifecycle(
    e16_operators,
    entrypoint,
):
    """An ID-free forward must not return state that its backward cannot use."""

    values = _make_expert_major_case(e16_operators, 1)
    unsupported = SonicMoE(
        replace(e16_operators.config, top_k=2),
        e16_operators.materialized.weights,
    )
    metadata = (
        values.expert_counts
        if "counts" in entrypoint
        else values.expert_offsets
    )

    with pytest.raises(
        NotImplementedError,
        match="complete bias-free BF16 SwiGLU H2048/I768/E16/top_k=1",
    ):
        getattr(unsupported, entrypoint)(
            values.hidden,
            metadata,
            values.route_weights,
        )
