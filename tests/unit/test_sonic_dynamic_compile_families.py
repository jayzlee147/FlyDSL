# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""CPU-only regression tests for dynamic E16 compile-family stability."""

from __future__ import annotations

import ast
import inspect
import math
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kernels.moe import sonic as sonic_module
from kernels.moe import sonic_backward as sonic_backward_module
from kernels.moe import sonic_grouped_tn as grouped_tn_module
from kernels.moe.sonic import (
    SonicMoE,
    SonicMoEConfig,
    SonicMoEDynamicWorkspacePool,
    SonicMoEWorkspace,
    warmup_sonic_e16_training,
)
from kernels.moe.sonic_dynamic_policy import (
    e16_route_policy_representative,
    select_e16_route_policy,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_e16_warmup_executes_each_finite_family_with_one_small_probe(
    monkeypatch,
):
    """Warmup must launch real forward/backward calls, not exact-R factories."""

    operator = object.__new__(SonicMoE)
    operator.config = _qwen3_e16_config()
    operator.weights = SimpleNamespace(
        weight_dtype="bf16",
        has_bias=False,
        device=torch.device("cuda", 0),
    )
    forward_calls = []
    backward_calls = []

    def fake_forward(*args, **kwargs):
        forward_calls.append((args, kwargs))
        return object(), SimpleNamespace(
            route_policy_size=kwargs["route_policy_size"]
        )

    operator.forward_expert_major_counts_training = fake_forward
    monkeypatch.setattr(
        sonic_module,
        "sonic_moe_backward_expert_major",
        lambda *args, **kwargs: backward_calls.append((args, kwargs)),
    )
    fake_tensor = _FakeTensor((16, 2048), torch.bfloat16)
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: fake_tensor)
    monkeypatch.setattr(torch, "arange", lambda *args, **kwargs: _FakeTensor((args[0],), kwargs["dtype"]))
    monkeypatch.setattr(torch, "ones", lambda *args, **kwargs: _FakeTensor((args[0],), kwargs["dtype"]))
    monkeypatch.setattr(torch, "zeros_like", lambda _tensor: fake_tensor)
    synchronizations = []
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device=None: synchronizations.append(device),
    )

    warmed = warmup_sonic_e16_training(
        operator,
        object(),
        object(),
        route_policy_sizes=(1, 4097, 20000, 32769, 134000),
    )

    assert warmed == (4096, 16384, 32768, 65536)
    assert [call[1]["route_policy_size"] for call in forward_calls] == list(warmed)
    assert all(call[0][0].shape == (16, 2048) for call in forward_calls)
    assert all(call[0][1].shape == (16,) for call in forward_calls)
    assert all(len(call[0]) == 3 for call in forward_calls)
    assert len(backward_calls) == len(warmed)
    assert all(call[1]["forward_state"].route_policy_size == policy for call, policy in zip(backward_calls, warmed))
    assert all("route_policy_size" not in call[1] for call in backward_calls)
    assert synchronizations == [torch.device("cuda", 0)]


@pytest.mark.parametrize(
    ("route_policy_size", "actual_routes"),
    (
        (2048, (2047, 2048, 2049)),
        (4096, (2049, 4095, 4096)),
        (8192, (8191, 8192, 8193)),
        (16384, (8193, 16383, 16384)),
        (32768, (16385, 32767, 32768)),
        (65536, (65535, 65536, 65537)),
        (134000, (133999, 134000, 134001)),
    ),
)
def test_forward_compile_family_uses_shared_policy_not_actual_routes(
    monkeypatch,
    route_policy_size,
    actual_routes,
):
    """Adjacent local-R values may change grids, but not compiled launchers."""

    config = _qwen3_e16_config()
    operator = SimpleNamespace(
        config=config,
        weights=SimpleNamespace(
            has_bias=False,
            device=torch.device("cuda", 0),
        ),
    )
    compile_fingerprints = []

    def fake_stage1(*args):
        compile_fingerprints.append(("stage1", args))
        return "stage1"

    def fake_stage2(*args):
        compile_fingerprints.append(("stage2", args))
        return "stage2"

    monkeypatch.setattr(
        sonic_module,
        "_get_stage1_training_launcher",
        fake_stage1,
    )
    monkeypatch.setattr(sonic_module, "_get_stage2_launcher", fake_stage2)

    per_route_fingerprints = []
    for routes in actual_routes:
        max_padded = routes + config.num_experts * (config.route_tile_m - 1)
        workspace = SimpleNamespace(
            tokens=routes,
            routes=routes,
            max_padded_tokens=max_padded,
            stage2_max_m_blocks=max_padded // config.stage2_tile_m,
            route_output=None,
        )
        start = len(compile_fingerprints)
        plan = SonicMoE._prepare_grouped_gemms_training(
            operator,
            workspace,
            interleaved_w1=False,
            token_indices_identity=True,
            route_policy_size=route_policy_size,
        )
        per_route_fingerprints.append(tuple(compile_fingerprints[start:]))
        expected_policy = select_e16_route_policy(routes, route_policy_size)
        assert plan.route_policy_size == e16_route_policy_representative(
            expected_policy
        )

    assert per_route_fingerprints[0] == per_route_fingerprints[1]
    assert per_route_fingerprints[1] == per_route_fingerprints[2]


class _FakeTensor:
    def __init__(self, shape, dtype, *, device=None):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.device = torch.device("cuda", 0) if device is None else device
        self.recorded_streams = []

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return 4 if self.dtype == torch.int32 else 2

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return id(self)

    def record_stream(self, stream):
        self.recorded_streams.append(stream)


def test_inference_compile_family_uses_shared_policy_not_actual_routes(
    monkeypatch,
):
    """Expert-major inference must not select cache/pipeline code by local R."""

    config = replace(_qwen3_e16_config(), stage2_pipeline_stages=None)
    device = torch.device("cuda", 0)
    fake_weight = _FakeTensor((1,), torch.bfloat16, device=device)
    operator = SimpleNamespace(
        config=config,
        weights=SimpleNamespace(
            weight_dtype="bf16",
            has_bias=False,
            gate_up=fake_weight,
            down=fake_weight,
            dummy_scale=fake_weight,
            gate_up_scale=None,
            down_scale=None,
            stage1_bias=None,
            stage2_bias=None,
        ),
    )
    compile_fingerprints = []

    def fake_stage1(*args):
        compile_fingerprints.append(("stage1", args))
        return "stage1"

    def fake_stage2(*args):
        compile_fingerprints.append(("stage2", args))
        return "stage2"

    monkeypatch.setattr(sonic_module, "_get_stage1_launcher", fake_stage1)
    monkeypatch.setattr(sonic_module, "_get_stage2_launcher", fake_stage2)
    monkeypatch.setattr(sonic_module, "_run_compiled", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device=None: object())

    per_route_fingerprints = []
    for routes in (15, 16, 17, 1023, 1024, 1025, 2047, 2048, 2049):
        max_padded = 128 * min(16, routes)
        workspace = SimpleNamespace(
            tokens=routes,
            routes=routes,
            max_padded_tokens=max_padded,
            stage1_max_m_blocks=max_padded // config.tile_m,
            stage2_max_m_blocks=max_padded // config.stage2_tile_m,
            sorted_expert_ids=_FakeTensor((16,), torch.int32, device=device),
            num_valid_ids=_FakeTensor((2,), torch.int32, device=device),
            sorted_token_ids=_FakeTensor((max(1, max_padded),), torch.int32, device=device),
            sorted_weights=_FakeTensor((max(1, max_padded),), torch.float32, device=device),
            intermediate=_FakeTensor(
                (max(1, max_padded), config.intermediate_size),
                torch.bfloat16,
                device=device,
            ),
            route_output=None,
        )
        hidden = _FakeTensor((routes, config.hidden_size), torch.bfloat16, device=device)
        output = _FakeTensor((routes, config.hidden_size), torch.bfloat16, device=device)
        start = len(compile_fingerprints)
        SonicMoE._run_grouped_gemms(
            operator,
            hidden,
            workspace,
            output,
            token_indices_identity=True,
            route_policy_size=4096,
        )
        per_route_fingerprints.append(tuple(compile_fingerprints[start:]))

    assert all(
        fingerprint == per_route_fingerprints[0]
        for fingerprint in per_route_fingerprints[1:]
    )


def test_grouped_tn_runtime_hot_cutoff_does_not_change_compile_family(
    monkeypatch,
):
    """The numeric ceil(R*11/64) cutoff must remain a runtime scalar."""

    compile_calls = []
    runtime_calls = []

    def fake_compile(*args, **kwargs):
        compile_calls.append((args, kwargs))
        return "compiled-grouped-tn"

    monkeypatch.setattr(grouped_tn_module, "compile_grouped_tn", fake_compile)
    monkeypatch.setattr(
        grouped_tn_module,
        "grouped_tn_launch_grid",
        lambda *_args: 256,
    )
    monkeypatch.setattr(
        grouped_tn_module,
        "_run_compiled",
        lambda *args: runtime_calls.append(args),
    )

    device = torch.device("cuda", 0)
    lhs = _FakeTensor((256, 2048), torch.bfloat16, device=device)
    rhs = _FakeTensor((256, 768), torch.bfloat16, device=device)
    frequency = _FakeTensor((16,), torch.int32, device=device)
    queue = _FakeTensor((1 + 2 * 16,), torch.int32, device=device)
    hot_queue = _FakeTensor((1 + 3 * 16,), torch.int32, device=device)
    output = _FakeTensor((16, 2048, 768), torch.bfloat16, device=device)
    stream = object()
    route_counts = (8192, 8193, 9000, 32769, 65536, 134000)
    runtime_cutoffs = tuple(
        sonic_backward_module._e16_dw2_hot_profile_min_rows(routes)
        for routes in route_counts
    )

    for cutoff in runtime_cutoffs:
        grouped_tn_module.grouped_tn_from_queue_flydsl(
            lhs,
            rhs,
            frequency,
            queue,
            output,
            block_m=256,
            block_n=256,
            block_k=64,
            k_padding=0,
            m_waves=4,
            n_waves=4,
            stages=2,
            min_expert_rows=cutoff,
            hot_expert_storage=hot_queue,
            hot_split_min_rows=cutoff,
            stream=stream,
        )

    assert len(set(runtime_cutoffs)) > 1
    assert all(call == compile_calls[0] for call in compile_calls)
    assert compile_calls[0][1]["exclude_hot_experts"] is True
    assert tuple(call[-6] for call in runtime_calls) == runtime_cutoffs
    assert tuple(call[-3] for call in runtime_calls) == runtime_cutoffs
    assert all(call[-4] is hot_queue for call in runtime_calls)


@pytest.mark.parametrize(
    ("routes", "expected"),
    (
        (0, False),
        (1, True),
        (17, True),
        (4095, True),
        (4096, True),
        (4097, True),
        (8191, True),
        (8192, True),
        (8193, True),
        (65536, True),
        (134000, True),
    ),
)
def test_standalone_e16_grouped_backward_has_no_exact_route_gate(
    routes,
    expected,
):
    assert (
        sonic_backward_module._use_e16_flat_grouped_backward(
            compute_dtype="bf16",
            activation="swiglu",
            hidden_size=2048,
            intermediate_size=768,
            num_experts=16,
            routes=routes,
            flat_routes=True,
            has_bias=False,
            reuse_forward_preactivation=False,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("route_policy_size", "expected_launches"),
    ((2048, 2), (8192, 1), (32768, 2), (65536, 2)),
)
def test_backward_balanced_and_hot_profiles_have_fixed_device_guard_structure(
    monkeypatch,
    route_policy_size,
    expected_launches,
):
    """Balanced/hot routing changes device guards, not host launch selection."""

    calls = []
    route_policy = select_e16_route_policy(
        route_policy_size,
        route_policy_size,
    )
    assert (
        sonic_backward_module._use_e16_dw1_dual_profile(
            direct_rhs=True,
            e16_flat_grouped=True,
            metadata_direct=False,
            route_policy=route_policy,
        )
        is (expected_launches == 2)
    )
    monkeypatch.setattr(
        sonic_backward_module,
        "grouped_tn_from_queue_flydsl",
        lambda *_args, **kwargs: calls.append(kwargs),
    )

    routes = route_policy_size
    sonic_backward_module._launch_grouped_dw2(
        object(),
        object(),
        SimpleNamespace(numel=lambda: 16),
        object(),
        object(),
        object(),
        object(),
        use_hostless_grouped=True,
        use_tn_metadata_direct=False,
        max_expert_rows=routes,
        route_policy=route_policy,
        hidden_size=2048,
        intermediate_size=768,
        active_experts=16,
        stream=object(),
    )

    assert len(calls) == expected_launches
    if expected_launches == 1:
        assert calls[0]["min_active_experts"] == 0
        assert calls[0]["max_active_experts"] is None
        assert calls[0]["min_expert_rows"] == 0
        assert calls[0]["max_expert_rows"] is None
        assert not calls[0]["active_guard_or_expert_rows"]
    else:
        hot_cutoff = sonic_backward_module._e16_dw2_hot_profile_min_rows(routes)
        sparse_expert_limit = (
            sonic_backward_module._E16_DW2_SMALL_TILE_MAX_ACTIVE_EXPERTS
        )
        assert calls[0]["min_active_experts"] == 0
        assert calls[0]["max_active_experts"] == sparse_expert_limit
        assert calls[0]["min_expert_rows"] == hot_cutoff
        assert calls[0]["max_expert_rows"] is None
        assert calls[0]["active_guard_or_expert_rows"]
        assert calls[1]["min_active_experts"] == sparse_expert_limit + 1
        assert calls[1]["max_active_experts"] is None
        assert calls[1]["min_expert_rows"] == 0
        assert calls[1]["max_expert_rows"] == hot_cutoff - 1
        assert not calls[1]["active_guard_or_expert_rows"]


def test_dw2_runtime_hot_cutoff_remains_adaptive_inside_one_policy(monkeypatch):
    """A shared tile family must not freeze the device-side load cutoff."""

    calls = []
    monkeypatch.setattr(
        sonic_backward_module,
        "grouped_tn_from_queue_flydsl",
        lambda *_args, **kwargs: calls.append(kwargs),
    )
    actual_routes = (8192, 9000, 134000)
    observed = []
    for routes in actual_routes:
        start = len(calls)
        sonic_backward_module._launch_grouped_dw2(
            object(),
            object(),
            SimpleNamespace(numel=lambda: 16),
            object(),
            object(),
            object(),
            object(),
            use_hostless_grouped=True,
            use_tn_metadata_direct=False,
            max_expert_rows=routes,
            route_policy=select_e16_route_policy(8192, 8192),
            hidden_size=2048,
            intermediate_size=768,
            active_experts=16,
            stream=object(),
        )
        invocation = calls[start:]
        assert len(invocation) == 1
        observed.append(invocation[0]["min_expert_rows"])

    assert observed == [0, 0, 0]

    # MEDIUM intentionally uses one minimax profile.  A dual-profile policy
    # keeps one device cutoff for every rank sharing that finite family.
    calls.clear()
    policy = select_e16_route_policy(32768, 32768)
    policy_cutoff = sonic_backward_module._e16_dw2_hot_profile_min_rows(
        e16_route_policy_representative(policy)
    )
    for routes in actual_routes:
        start = len(calls)
        sonic_backward_module._launch_grouped_dw2(
            object(),
            object(),
            SimpleNamespace(numel=lambda: 16),
            object(),
            object(),
            object(),
            object(),
            use_hostless_grouped=True,
            use_tn_metadata_direct=False,
            max_expert_rows=routes,
            route_policy=policy,
            hidden_size=2048,
            intermediate_size=768,
            active_experts=16,
            stream=object(),
        )
        invocation = calls[start:]
        assert len(invocation) == 2
        assert invocation[0]["min_expert_rows"] == policy_cutoff
        assert invocation[1]["max_expert_rows"] == policy_cutoff - 1


@pytest.mark.parametrize(
    "routes",
    (
        16385,
        32768,
        32769,
        65536,
        134000,
        1_000_000,
    ),
)
def test_hot_split_scratch_is_actual_sized_and_policy_bounded(routes):
    """Dynamic scratch is minimal while XLARGE retains a finite upper bound."""

    policy = select_e16_route_policy(routes)
    capacity, split_rows, min_hot_rows, split_activation_rows = (
        sonic_backward_module._e16_hot_split_schedule(routes, 16, policy)
    )
    required = grouped_tn_module.hot_split_descriptor_capacity(
        routes,
        16,
        split_rows,
        min_hot_rows,
    )

    policy_capacity = (
        sonic_backward_module._E16_DW1_LARGE_SPLIT_CAPACITY
        if policy.name == "LARGE"
        else sonic_backward_module._E16_DW1_XLARGE_SPLIT_CAPACITY
    )
    assert capacity == max(1, required)
    assert required <= capacity <= policy_capacity
    assert split_rows >= sonic_backward_module._E16_DW1_SPLIT_ROWS
    assert split_rows % sonic_backward_module._E16_DW1_SPLIT_ROW_QUANTUM == 0
    expected_min_hot_rows = split_rows + 1
    if policy.name == "XLARGE":
        expected_min_hot_rows = max(
            expected_min_hot_rows,
            sonic_backward_module._e16_dw2_hot_profile_min_rows(routes),
        )
    assert min_hot_rows == expected_min_hot_rows
    expected_activation_rows = max(
        min_hot_rows + 1,
        (
            split_rows
            * sonic_backward_module._E16_DW1_DENSE_SPLIT_TRIGGER_NUMERATOR
            + sonic_backward_module._E16_DW1_DENSE_SPLIT_TRIGGER_DENOMINATOR
            - 1
        )
        // sonic_backward_module._E16_DW1_DENSE_SPLIT_TRIGGER_DENOMINATOR,
    )
    assert split_activation_rows == expected_activation_rows
    # dW1 FP32 partials stay bounded at 384 MiB for [32, 1536, 2048]
    # while smaller invocations reserve proportionally less.
    assert capacity * 1536 * 2048 * 4 <= 384 * 1024 * 1024


def test_standalone_hot_split_builder_thresholds_are_runtime_scalars():
    """The reusable helper must not recompile for every dynamic route count."""

    parameters = inspect.signature(
        grouped_tn_module.compile_hot_split_queues
    ).parameters

    assert "split_rows" not in parameters
    assert "min_hot_rows" not in parameters


def test_standalone_hot_split_builder_forwards_dynamic_thresholds(monkeypatch):
    """Changing split thresholds changes arguments, not the compile request."""

    compile_calls = []
    runtime_calls = []

    def fake_compile(*args, **kwargs):
        compile_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(
        grouped_tn_module,
        "compile_hot_split_queues",
        fake_compile,
    )
    monkeypatch.setattr(
        grouped_tn_module,
        "_run_compiled",
        lambda *args: runtime_calls.append(args),
    )

    routes = 1024
    experts = 4
    device = torch.device("cuda", 0)
    active_queue = _FakeTensor((1 + 2 * experts,), torch.int32, device=device)
    cold_queue = _FakeTensor((1 + 2 * experts,), torch.int32, device=device)
    split_queue = _FakeTensor((1 + 3 * 32,), torch.int32, device=device)
    hot_queue = _FakeTensor((1 + 3 * experts,), torch.int32, device=device)
    frequency = _FakeTensor((experts,), torch.int32, device=device)
    stream = object()
    thresholds = ((64, 128), (128, 256), (256, 512))

    for split_rows, min_hot_rows in thresholds:
        grouped_tn_module.build_hot_split_queues_flydsl(
            frequency,
            active_queue,
            routes=routes,
            split_rows=split_rows,
            min_hot_rows=min_hot_rows,
            cold_queue=cold_queue,
            split_queue=split_queue,
            hot_queue=hot_queue,
            stream=stream,
        )

    assert compile_calls == [compile_calls[0]] * len(thresholds)
    assert compile_calls[0][0] == (experts, 0)
    assert tuple(call[-3:-1] for call in runtime_calls) == thresholds


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    module = ast.parse(path.read_text())
    return next(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _method_node(path: Path, class_name: str, name: str) -> ast.FunctionDef:
    module = ast.parse(path.read_text())
    owner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


@pytest.mark.parametrize(
    "method_name",
    (
        "_forward_routes_on_current_device",
        "_forward_routes_training_on_current_device",
    ),
)
def test_all_flat_route_entrypoints_use_growable_invocation_owned_workspace(
    method_name,
):
    """Generic ragged R jitter must not recreate exact-shape workspaces."""

    function = _method_node(
        _REPO_ROOT / "kernels/moe/sonic.py",
        "SonicMoE",
        method_name,
    )
    method_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    self_calls = {
        call.func.attr
        for call in method_calls
        if isinstance(call.func.value, ast.Name) and call.func.value.id == "self"
    }
    validate_out = next(
        call for call in method_calls if call.func.attr == "_validate_out"
    )
    invocation_owned = next(
        keyword.value
        for keyword in validate_out.keywords
        if keyword.arg == "invocation_owned_default"
    )

    assert "reserve_dynamic_routes" in self_calls
    assert "reserve" not in self_calls
    assert isinstance(invocation_owned, ast.Constant)
    assert invocation_owned.value is True


def test_sorter_runtime_routes_and_partitions_are_absent_from_cache_key():
    """Static guard against putting dynamic R/P back in the sorter JIT key."""

    function = _function_node(
        _REPO_ROOT / "kernels/moe/moe_ragged_sorting_kernel.py",
        "moe_expert_major_sorting_flydsl",
    )
    cache_value = next(
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "cache_key"
            for target in node.targets
        )
    )
    key_names = {
        node.id for node in ast.walk(cache_value) if isinstance(node, ast.Name)
    }

    assert key_names.isdisjoint(
        {"routes", "partition_routes", "identity_partitions", "route_policy_size"}
    )


def test_fused_metadata_stage1_launcher_has_no_runtime_route_key():
    """The fused Stage-1 launcher is compiled by policy, not exact R/P."""

    function = _function_node(
        _REPO_ROOT / "kernels/moe/sonic.py",
        "_get_stage1_training_launcher",
    )
    parameter_names = {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }

    assert "routes" not in parameter_names
    assert "route_policy_size" not in parameter_names
    assert "identity_partitions" not in parameter_names


def _cpu_flat_capacity(
    config: SonicMoEConfig,
    tokens: int,
    routes: int,
) -> SonicMoEWorkspace:
    active_experts = min(config.num_experts, routes)
    max_blocks = (
        routes + active_experts * (config.route_tile_m - 1)
    ) // config.route_tile_m
    max_padded = max_blocks * config.route_tile_m
    return SonicMoEWorkspace(
        tokens=tokens,
        routes=routes,
        route_tile_m=config.route_tile_m,
        max_padded_tokens=max_padded,
        max_m_blocks=max_blocks,
        stage1_max_m_blocks=max_padded // config.tile_m,
        stage2_max_m_blocks=max_padded // config.stage2_tile_m,
        sorted_token_ids=torch.empty(max(1, max_padded), dtype=torch.int32),
        sorted_route_ids=torch.empty(max(1, max_padded), dtype=torch.int32),
        sorted_weights=torch.empty(max(1, max_padded), dtype=torch.float32),
        sorted_expert_ids=torch.empty(max(1, max_blocks), dtype=torch.int32),
        num_valid_ids=torch.empty(2, dtype=torch.int32),
        sorting_workspace=torch.empty(config.num_experts, dtype=torch.int32),
        expert_frequency=torch.empty(config.num_experts, dtype=torch.int32),
        router_topk_weights=torch.empty(
            (tokens, config.top_k),
            dtype=torch.float32,
        ),
        router_topk_ids=torch.empty(
            (tokens, config.top_k),
            dtype=torch.int32,
        ),
        router_topk_expert_indices=torch.empty(
            (tokens, config.top_k),
            dtype=torch.int32,
        ),
        intermediate=torch.empty(
            (max(1, max_padded), config.intermediate_size),
            dtype=torch.bfloat16,
        ),
        route_output=None,
        output=None,
    )


def test_shared_workspace_pool_returns_empty_active_view_without_zero_storage(
    monkeypatch,
):
    """T=R=0 keeps safe backing buffers while exposing exact active extents."""

    config = _qwen3_e16_config()
    allocations = []

    def fake_allocate(
        cls,
        allocation_config,
        tokens,
        _device,
        *,
        routes=None,
        reusable_output=True,
    ):
        assert allocation_config == config
        assert routes is not None
        assert not reusable_output
        capacity = _cpu_flat_capacity(allocation_config, tokens, routes)
        allocations.append(capacity)
        return capacity

    fake_stream = SimpleNamespace(cuda_stream=1234)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda _device=None: fake_stream,
    )
    monkeypatch.setattr(
        SonicMoEWorkspace,
        "allocate",
        classmethod(fake_allocate),
    )

    pool = SonicMoEDynamicWorkspacePool()
    first = pool.reserve(config, 0, 0, torch.device("cuda", 0))
    second = pool.reserve(config, 0, 0, torch.device("cuda", 0))
    entry = next(iter(pool._entries.values()))

    assert len(allocations) == 1
    assert len(pool) == 1
    assert entry.capacity is allocations[0]
    assert first._launch_lock is second._launch_lock is entry.launch_lock
    for view in (first, second):
        assert (view.tokens, view.routes) == (0, 0)
        assert view.max_padded_tokens == 0
        assert view.max_m_blocks == 0
        assert view.stage1_max_m_blocks == 0
        assert view.stage2_max_m_blocks == 0
        assert tuple(view.router_topk_weights.shape) == (0, 1)
        assert tuple(view.router_topk_ids.shape) == (0, 1)
        assert tuple(view.router_topk_expert_indices.shape) == (0, 1)
        assert view.sorted_token_ids.numel() == 1
        assert view.sorted_route_ids is not None
        assert view.sorted_route_ids.numel() == 1
        assert view.sorted_weights.numel() == 1
        assert view.sorted_expert_ids.numel() == 1
        assert tuple(view.intermediate.shape) == (1, 768)
        assert view.output is None


def test_active_expert_queue_compiler_has_no_metadata_extent_key():
    """Standalone grouped-TN queue construction must not compile per R."""

    function = _function_node(
        _REPO_ROOT / "kernels/moe/sonic_grouped_tn.py",
        "compile_active_expert_queue",
    )
    parameter_names = {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }

    assert "routes" not in parameter_names
    assert "max_metadata_blocks" not in parameter_names


def test_retained_identity_dx_extent_uses_only_allocated_16bit_buffer():
    """The largest retained E16 state must not inherit an unused FP32 limit."""

    hidden_size = 2048
    retained_state_high_water = ((1 << 31) - 1) // (2 * 768 * 2)
    fp32_accum_high_water = ((1 << 32) - 1) // (hidden_size * 4)

    assert retained_state_high_water == 699050
    assert fp32_accum_high_water == 524287
    assert retained_state_high_water > fp32_accum_high_water

    # Identity/segmented retained paths write a real BF16 dX and no FP32
    # scatter workspace, so the retained-state high-water mark is legal.
    sonic_backward_module._validate_backward_dx_extent(
        retained_state_high_water,
        hidden_size,
        requires_fp32_accum=False,
    )

    # The generic arbitrary-order scatter still enforces its actual FP32
    # allocation boundary exactly.
    sonic_backward_module._validate_backward_dx_extent(
        fp32_accum_high_water,
        hidden_size,
        requires_fp32_accum=True,
    )
    with pytest.raises(ValueError, match="FP32 input-gradient workspace"):
        sonic_backward_module._validate_backward_dx_extent(
            fp32_accum_high_water + 1,
            hidden_size,
            requires_fp32_accum=True,
        )
