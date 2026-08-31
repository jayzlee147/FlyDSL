# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""device runtime registry and compile-backend pairing."""

import threading

import pytest

import flydsl.runtime.device_runtime as dr
import flydsl.runtime.device_runtime.rocm as rocm_runtime

pytestmark = [pytest.mark.l0_backend_agnostic]


@pytest.fixture(autouse=True)
def _reset_device_runtime_singleton():
    """Each test starts without a cached DeviceRuntime instance."""
    dr._instance = None
    dr._runtime_cls_override = None
    dr._EXTRA_MAPPINGS.clear()
    yield
    dr._instance = None
    dr._runtime_cls_override = None
    dr._EXTRA_MAPPINGS.clear()


class _FakeCudaRuntime(dr.DeviceRuntime):
    kind = "cuda"

    def device_count(self) -> int:
        return 1

    def current_device_id(self) -> int:
        return 0


def test_default_runtime_kind_stays_rocm(monkeypatch):
    """Community users that do not opt into another runtime keep the ROCm default."""
    monkeypatch.delenv("FLYDSL_COMPILE_BACKEND", raising=False)
    monkeypatch.delenv("FLYDSL_RUNTIME_KIND", raising=False)
    rt = dr.get_device_runtime()
    assert rt.kind == "rocm"


def test_default_compile_runtime_pairing_does_not_need_env(monkeypatch):
    monkeypatch.delenv("FLYDSL_COMPILE_BACKEND", raising=False)
    monkeypatch.delenv("FLYDSL_RUNTIME_KIND", raising=False)
    dr.ensure_compile_runtime_pairing_from_env("rocm")
    assert dr._instance is None


def test_rocm_runtime_kind_matches_compile_backend(monkeypatch):
    monkeypatch.delenv("FLYDSL_RUNTIME_KIND", raising=False)
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    rt = dr.get_device_runtime()
    assert rt.kind == "rocm"
    dr.ensure_compile_runtime_compatible("rocm", runtime=rt)


def test_rocm_current_device_loads_rocm_7_soname(monkeypatch):
    """ROCm 7 images may install only the versioned ``.so.7`` runtime."""

    class _FakeHipGetDevice:
        argtypes = None
        restype = None

        def __call__(self, device_ptr):
            device_ptr._obj.value = 3
            return 0

    class _FakeHipLibrary:
        hipGetDevice = _FakeHipGetDevice()

    attempted = []

    def _load_library(soname):
        attempted.append(soname)
        if soname != "libamdhip64.so.7":
            raise OSError(soname)
        return _FakeHipLibrary()

    monkeypatch.setattr(rocm_runtime, "_HIP_LIB", None)
    monkeypatch.setattr(rocm_runtime, "_HIP_LIB_TRIED", False)
    monkeypatch.setattr(rocm_runtime.ctypes, "CDLL", _load_library)

    assert rocm_runtime._hip_get_device() == 3
    assert attempted == ["libamdhip64.so", "libamdhip64.so.7"]


def test_rocm_current_device_initializes_library_once_across_threads(monkeypatch):
    class _FakeHipGetDevice:
        argtypes = None
        restype = None

        def __call__(self, device_ptr):
            device_ptr._obj.value = 3
            return 0

    class _FakeHipLibrary:
        hipGetDevice = _FakeHipGetDevice()

    first_load_started = threading.Event()
    release_first_load = threading.Event()
    second_lock_attempted = threading.Event()
    second_finished = threading.Event()
    attempted = []
    results = []
    errors = []

    class _TrackingLock:
        def __init__(self):
            self._lock = threading.Lock()

        def __enter__(self):
            if threading.current_thread().name == "hip-second":
                second_lock_attempted.set()
            self._lock.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._lock.release()

    def _load_library(soname):
        attempted.append(soname)
        if soname == "libamdhip64.so":
            first_load_started.set()
            if not release_first_load.wait(timeout=5):
                raise TimeoutError("HIP library probe was not released")
            raise OSError(soname)
        if soname == "libamdhip64.so.7":
            return _FakeHipLibrary()
        raise OSError(soname)

    def _get_device():
        try:
            results.append(rocm_runtime._hip_get_device())
        except Exception as error:
            errors.append(error)
        finally:
            if threading.current_thread().name == "hip-second":
                second_finished.set()

    monkeypatch.setattr(rocm_runtime, "_HIP_LIB", None)
    monkeypatch.setattr(rocm_runtime, "_HIP_LIB_TRIED", False)
    monkeypatch.setattr(rocm_runtime, "_HIP_LIB_LOCK", _TrackingLock())
    monkeypatch.setattr(rocm_runtime.ctypes, "CDLL", _load_library)

    first = threading.Thread(target=_get_device, name="hip-first")
    second = threading.Thread(target=_get_device, name="hip-second")
    first.start()
    try:
        assert first_load_started.wait(timeout=5)
        assert rocm_runtime._HIP_LIB_TRIED is False
        second.start()
        assert second_lock_attempted.wait(timeout=5)
        assert not second_finished.is_set()
        assert rocm_runtime._HIP_LIB_TRIED is False
    finally:
        release_first_load.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not errors
    assert results == [3, 3]
    assert attempted == ["libamdhip64.so", "libamdhip64.so.7"]
    assert rocm_runtime._HIP_LIB_TRIED is True
    assert isinstance(rocm_runtime._HIP_LIB, _FakeHipLibrary)


def test_ensure_mismatch_raises():
    bad = _FakeCudaRuntime()
    with pytest.raises(RuntimeError, match="requires device runtime kind"):
        dr.ensure_compile_runtime_compatible("rocm", runtime=bad)


def test_unknown_runtime_kind_env(monkeypatch):
    """Invalid FLYDSL_RUNTIME_KIND fails at compile/runtime pairing first."""
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "not_a_real_kind")
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    with pytest.raises(RuntimeError, match="requires device runtime kind"):
        dr.get_device_runtime()


def test_unknown_runtime_kind_after_pairing_passes(monkeypatch):
    """When env strings agree with mapping, unknown kind fails in class lookup."""
    dr.register_compile_runtime_mapping("custom", "weird_kind")
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "custom")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "weird_kind")
    try:
        with pytest.raises(ValueError, match="Unknown FLYDSL_RUNTIME_KIND"):
            dr.get_device_runtime()
    finally:
        dr._EXTRA_MAPPINGS.pop("custom", None)


def test_register_compile_runtime_mapping():
    dr.register_compile_runtime_mapping("foo", "rocm")
    try:
        dr.ensure_compile_runtime_compatible("foo", runtime=dr.RocmDeviceRuntime())
    finally:
        dr._EXTRA_MAPPINGS.pop("foo", None)


def test_pairing_from_env_no_singleton(monkeypatch):
    """Pairing check used on compile path must not create DeviceRuntime."""
    monkeypatch.delenv("FLYDSL_RUNTIME_KIND", raising=False)
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    dr.ensure_compile_runtime_pairing_from_env("rocm")
    assert dr._instance is None


def test_pairing_from_env_mismatch_raises(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "not_a_registered_kind")
    with pytest.raises(RuntimeError, match="requires device runtime kind"):
        dr.ensure_compile_runtime_pairing_from_env("rocm")
