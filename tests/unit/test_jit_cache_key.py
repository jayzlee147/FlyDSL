# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.compiler.jit_function as jit_function_module
import flydsl.expr as fx
import flydsl.runtime.device_runtime as device_runtime_module
from flydsl.compiler.jit_argument import JitArgumentRegistry
from flydsl.compiler.jit_executor import CompiledArtifact
from flydsl.compiler.kernel_function import CompilationContext


class _FakeCudaStream:
    cuda_stream = 1234


JitArgumentRegistry.register(_FakeCudaStream)(fx.Stream)


@flyc.jit
def _stream_launch(stream: fx.Stream = fx.Stream(None)):
    pass


@flyc.jit
def _constexpr_launch(value: fx.Constexpr[int]):
    pass


@flyc.jit
def _runtime_int32_launch(n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    pass


def _cache_key(jit_fn, *args):
    jit_fn._ensure_sig()
    bound = jit_fn._sig.bind(*args)
    bound.apply_defaults()
    return jit_fn._resolve_and_make_cache_key(bound.arguments)


def _full_cache_key(jit_fn, *args):
    jit_fn._ensure_sig()
    bound = jit_fn._sig.bind(*args)
    bound.apply_defaults()
    return jit_fn._build_full_cache_key(bound.arguments)


def test_stream_cache_key_ignores_runtime_representation():
    """CPU AOT can use raw 0 while GPU runtime passes a stream object."""
    keys = [
        _cache_key(_stream_launch),
        _cache_key(_stream_launch, 0),
        _cache_key(_stream_launch, fx.Stream(0)),
        _cache_key(_stream_launch, _FakeCudaStream()),
    ]

    assert keys[0] == keys[1] == keys[2] == keys[3]
    assert ("stream", (fx.Stream,)) in keys[0]


def test_full_cache_key_is_portable_for_every_stream_representation(monkeypatch):
    """Artifact identity is device-free; execution identity is device-local."""

    stream_args = ((), (0,), (fx.Stream(0),), (_FakeCudaStream(),))

    monkeypatch.setattr(jit_function_module, "_current_device_cache_signature", lambda: ("rocm", 0))
    artifact_keys0 = [_full_cache_key(_stream_launch, *args) for args in stream_args]
    execution_keys0 = [_stream_launch._build_execution_cache_key(key) for key in artifact_keys0]

    monkeypatch.setattr(jit_function_module, "_current_device_cache_signature", lambda: ("rocm", 1))
    artifact_keys1 = [_full_cache_key(_stream_launch, *args) for args in stream_args]
    execution_keys1 = [_stream_launch._build_execution_cache_key(key) for key in artifact_keys1]

    assert len(set(artifact_keys0 + artifact_keys1)) == 1
    assert all(name != "_device_" for name, _value in artifact_keys0[0])
    assert len(set(execution_keys0)) == 1
    assert len(set(execution_keys1)) == 1
    assert execution_keys0[0] != execution_keys1[0]
    assert execution_keys0[0][0] == ("_device_", ("rocm", 0))
    assert execution_keys1[0][0] == ("_device_", ("rocm", 1))


def test_full_cache_key_is_portable_without_stream_argument(monkeypatch):
    monkeypatch.setattr(jit_function_module, "_current_device_cache_signature", lambda: ("rocm", 0))
    artifact_key0 = _full_cache_key(_constexpr_launch, 7)
    execution_key0 = _constexpr_launch._build_execution_cache_key(artifact_key0)

    monkeypatch.setattr(jit_function_module, "_current_device_cache_signature", lambda: ("rocm", 1))
    artifact_key1 = _full_cache_key(_constexpr_launch, 7)
    execution_key1 = _constexpr_launch._build_execution_cache_key(artifact_key1)

    assert artifact_key0 == artifact_key1
    assert all(name != "_device_" for name, _value in artifact_key0)
    assert execution_key0 != execution_key1


def test_current_device_cache_signature_uses_active_runtime(monkeypatch):
    class _FakeRuntime:
        kind = "test"

        @staticmethod
        def current_device_id():
            return 3

    monkeypatch.setattr(device_runtime_module, "get_device_runtime", lambda: _FakeRuntime())
    assert jit_function_module._current_device_cache_signature() == ("test", 3)


def test_compiled_artifact_clone_is_unmaterialized_and_preserves_processors():
    class _FakeModule:
        def __str__(self):
            return "module { func.func @launch() }"

    processor = lambda module: module  # noqa: E731 - deliberately unpicklable
    artifact = CompiledArtifact(
        _FakeModule(),
        "launch",
        "source ir",
        post_load_processors=[processor],
        link_libs=["libexample.so"],
        uses_explicit_module=True,
    )
    artifact._module = object()
    artifact._engine = object()
    artifact._jit_module = object()
    artifact._func_exe = object()
    artifact._ctx = object()

    clone = artifact.clone_unmaterialized()

    assert clone is not artifact
    assert clone.ir == artifact.ir
    assert clone.source_ir == artifact.source_ir
    assert clone._entry == artifact._entry
    assert clone._post_load_processors == [processor]
    assert clone._post_load_processors is not artifact._post_load_processors
    assert clone._post_load_processors[0] is processor
    assert clone._link_libs == ["libexample.so"]
    assert clone._link_libs is not artifact._link_libs
    assert clone._uses_explicit_module is True
    assert clone._module is None
    assert clone._engine is None
    assert clone._jit_module is None
    assert clone._func_exe is None
    assert not hasattr(clone, "_ctx")
    assert clone._lock is not artifact._lock


def test_compile_only_never_resolves_current_device(monkeypatch):
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "0")
    monkeypatch.setattr(jit_function_module, "_flydsl_key", lambda: "test-flydsl-key")

    def fail_if_device_is_resolved():
        raise AssertionError("compile-only must not resolve the current runtime device")

    monkeypatch.setattr(
        jit_function_module,
        "_current_device_cache_signature",
        fail_if_device_is_resolved,
    )

    def compile_noop(cls, module, *, arch: str = "", func_name: str = "", link_libs=None):
        return module

    monkeypatch.setattr(jit_function_module.MlirCompiler, "compile", classmethod(compile_noop))

    @flyc.jit
    def launch():
        pass

    assert launch() is None
    assert len(launch._artifact_cache) == 1
    portable = next(iter(launch._artifact_cache.values()))
    assert launch._last_compiled[1] is portable
    assert launch._mem_cache == {}
    assert launch._call_state_cache == {}


def test_execution_cache_clones_one_portable_artifact_per_device(monkeypatch):
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("COMPILE_ONLY", "0")
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "0")
    monkeypatch.setattr(jit_function_module, "_flydsl_key", lambda: "test-flydsl-key")

    active_device = {"id": 0}
    monkeypatch.setattr(
        jit_function_module,
        "_current_device_cache_signature",
        lambda: ("rocm", active_device["id"]),
    )

    compile_calls = []

    def compile_noop(cls, module, *, arch: str = "", func_name: str = "", link_libs=None):
        compile_calls.append(module)
        return module

    monkeypatch.setattr(jit_function_module.MlirCompiler, "compile", classmethod(compile_noop))

    materialized = []

    def fake_get_func_exe(self):
        materialized.append(self)
        return self

    monkeypatch.setattr(CompiledArtifact, "_get_func_exe", fake_get_func_exe)

    built_states = []

    class _FakeCallState:
        def __init__(self, artifact):
            self.artifact = artifact
            self.calls = 0

        def __call__(self, args_tuple):
            self.calls += 1
            return self

    def fake_build_call_state(sig, args_tuple, artifact):
        state = _FakeCallState(artifact)
        built_states.append(state)
        return state

    monkeypatch.setattr(jit_function_module, "_build_call_state", fake_build_call_state)

    @flyc.jit
    def launch():
        pass

    device0_state = launch()
    active_device["id"] = 1
    device1_state = launch()
    active_device["id"] = 0
    assert launch() is device0_state

    assert device0_state is not device1_state
    assert len(compile_calls) == 1
    assert len(launch._artifact_cache) == 1
    assert len(launch._mem_cache) == 2
    assert len(launch._call_state_cache) == 2
    portable = next(iter(launch._artifact_cache.values()))
    assert portable not in materialized
    assert set(materialized) == set(launch._mem_cache.values())
    assert device0_state.artifact is not portable
    assert device1_state.artifact is not portable
    assert device0_state.calls == 2
    assert device1_state.calls == 1
    assert len(built_states) == 2


def test_constexpr_values_still_participate_in_cache_key():
    assert _cache_key(_constexpr_launch, 1) != _cache_key(_constexpr_launch, 2)


def test_future_annotations_runtime_int32_ignores_value_in_cache_key():
    """`from __future__ import annotations` stringifies fx.Int32; resolve_signature must eval it back so the value stays out of the cache key."""
    key1 = _cache_key(_runtime_int32_launch, 1)
    key2 = _cache_key(_runtime_int32_launch, 2)

    assert key1 == key2
    assert ("n", (fx.Int32,)) in key1
    assert ("n", (int, 1)) not in key1


def test_thread_local_wpe_overrides_persistent_hint_and_enters_cache_key():
    @flyc.jit
    def launch(stream: fx.Stream = fx.Stream(None)):
        pass

    launch.compile_hints = {"waves_per_eu": 4}
    persistent = _cache_key(launch)
    with CompilationContext.compile_hints({"waves_per_eu": 1}):
        outer = _cache_key(launch)
        assert launch._effective_compile_hints()["waves_per_eu"] == 1
        with CompilationContext.compile_hints({"waves_per_eu": 2}):
            inner = _cache_key(launch)
            assert launch._effective_compile_hints()["waves_per_eu"] == 2
        assert launch._effective_compile_hints()["waves_per_eu"] == 1
    assert launch._effective_compile_hints()["waves_per_eu"] == 4
    with CompilationContext.compile_hints({"waves_per_eu": "2"}):
        invalid_type = _cache_key(launch)

    assert len({persistent, outer, inner}) == 3
    assert invalid_type != inner
