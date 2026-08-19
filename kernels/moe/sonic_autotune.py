# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shape-bucket autotuning for the host-composed gfx950 SonicMoE forward."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import uuid
from dataclasses import replace
from pathlib import Path

import torch

import flydsl
from flydsl.autotune import do_bench
from flydsl.runtime.device import get_rocm_arch
from kernels.moe.moe_2stage_a16wmix.gemm1 import compile_gemm1_a16w4_port
from kernels.moe.moe_2stage_a16wmix.gemm2 import compile_gemm2_a16w4_port
from kernels.moe.moe_sorting_kernel import moe_softmax_sort_flydsl
from kernels.moe.sonic import SonicMoE, SonicMoEConfig, SonicMoEWeights


_CACHE_SCHEMA_VERSION = 2
_TUNING_FIELDS = (
    "tile_m",
    "tile_n",
    "tile_k",
    "down_tile_n",
    "down_tile_k",
    "stage1_b_cache_mod",
    "stage2_b_cache_mod",
    "stage1_xcd_swizzle",
    "stage2_xcd_swizzle",
    "waves_per_eu",
    "persistent_stage2",
)


def _config_tuning_dict(config: SonicMoEConfig) -> dict[str, int | bool | None]:
    return {name: getattr(config, name) for name in _TUNING_FIELDS}


def _source_fingerprint() -> str:
    source_files = (
        __file__,
        inspect.getsourcefile(compile_gemm1_a16w4_port),
        inspect.getsourcefile(compile_gemm2_a16w4_port),
        inspect.getsourcefile(moe_softmax_sort_flydsl),
        inspect.getsourcefile(SonicMoE),
    )
    paths = {Path(path).resolve() for path in source_files if path is not None}
    moe_dir = Path(__file__).resolve().parent
    repo_root = moe_dir.parents[1]
    paths.update(
        {
            moe_dir / "topk_gating_softmax_kernel.py",
            moe_dir / "moe_2stage_a16wmix" / "utils.py",
            repo_root / "kernels" / "common" / "buffer_ops.py",
            repo_root / "kernels" / "common" / "kernels_common.py",
            repo_root / "kernels" / "common" / "layout_utils.py",
            repo_root / "kernels" / "common" / "tensor_shim.py",
        }
    )
    digest = hashlib.sha256()
    for path in sorted(paths):
        if not path.is_file():
            continue
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _token_bucket(tokens: int) -> int:
    """Power-of-two ceiling used by SonicMoE's M-bucket dispatch cache."""

    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens}")
    return 1 << (tokens - 1).bit_length()


def default_sonic_moe_candidates(base: SonicMoEConfig) -> tuple[SonicMoEConfig, ...]:
    """Return a bounded gfx950 tile search space, pruning illegal configs."""

    tile_shapes = [
        (
            base.tile_n,
            base.tile_k,
            base.stage2_tile_n,
            base.stage2_tile_k,
        ),
        (128, 128, 128, 128),
        (256, 128, 256, 128),
        (128, 256, 128, 256),
    ]
    tile_shapes = list(dict.fromkeys(tile_shapes))
    candidates: list[SonicMoEConfig] = [base]
    seen = {
        (
            base.tile_m,
            base.tile_n,
            base.tile_k,
            base.stage2_tile_n,
            base.stage2_tile_k,
        )
    }
    for tile_m in (16, 32, 64, 128):
        for tile_n, tile_k, down_tile_n, down_tile_k in tile_shapes:
            try:
                candidate = replace(
                    base,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    tile_k=tile_k,
                    down_tile_n=down_tile_n,
                    down_tile_k=down_tile_k,
                )
            except (TypeError, ValueError):
                continue
            effective = (
                candidate.tile_m,
                candidate.tile_n,
                candidate.tile_k,
                candidate.stage2_tile_n,
                candidate.stage2_tile_k,
            )
            if effective not in seen:
                seen.add(effective)
                candidates.append(candidate)
    return tuple(candidates)


class SonicMoEAutotuner:
    """Benchmark complete router+sort+GEMM candidates and cache the winner.

    Prepared weights are shared by all tile configurations. Winners are keyed
    by a power-of-two token bucket, logical model shape, router dtype, weight
    format, gfx architecture, toolchain, candidate set, and kernel source hash.
    """

    def __init__(
        self,
        base_config: SonicMoEConfig,
        weights: SonicMoEWeights,
        *,
        candidates: tuple[SonicMoEConfig, ...] | list[SonicMoEConfig] | None = None,
        warmup: int = 5,
        rep: int = 20,
        cache_dir: str | os.PathLike[str] | None = None,
        validate_candidates: bool = True,
        profile_key: str | None = None,
    ):
        if warmup < 0 or rep <= 0:
            raise ValueError(f"warmup must be >= 0 and rep > 0, got {warmup}/{rep}")
        self.base_config = base_config
        self.weights = weights
        self.candidates = tuple(
            default_sonic_moe_candidates(base_config)
            if candidates is None
            else candidates
        )
        if not self.candidates:
            raise ValueError("at least one SonicMoE autotune candidate is required")
        self._validate_candidate_semantics()
        self.warmup = int(warmup)
        self.rep = int(rep)
        self.validate_candidates = bool(validate_candidates)
        if profile_key is not None and not isinstance(profile_key, str):
            raise TypeError(f"profile_key must be str or None, got {type(profile_key).__name__}")
        self.profile_key = profile_key
        root = (
            Path(cache_dir)
            if cache_dir is not None
            else Path(os.environ.get("FLYDSL_AUTOTUNE_CACHE_DIR", "~/.flydsl/autotune"))
        )
        self.cache_file = root.expanduser().resolve() / "sonic_moe.json"
        self._source_hash = _source_fingerprint()
        self._winner_cache: dict[str, SonicMoEConfig] = {}
        self._ops: dict[SonicMoEConfig, SonicMoE] = {}
        self.best_config: SonicMoEConfig | None = None
        self.last_results: tuple[tuple[SonicMoEConfig, float], ...] = ()
        self.search_count = 0
        self._load_disk_cache()

    @property
    def workspace(self):
        if self.best_config is None:
            return None
        return self._op_for(self.best_config).workspace

    def clear_workspace(self) -> None:
        for op in self._ops.values():
            op.clear_workspace()

    def _validate_candidate_semantics(self) -> None:
        semantic_fields = (
            "hidden_size",
            "intermediate_size",
            "num_experts",
            "top_k",
            "renormalize",
        )
        expected = tuple(getattr(self.base_config, name) for name in semantic_fields)
        SonicMoE(self.base_config, self.weights)
        for candidate in self.candidates:
            actual = tuple(getattr(candidate, name) for name in semantic_fields)
            if actual != expected:
                raise ValueError(
                    "autotune candidates may change tuning fields only; "
                    f"semantic values differ: {actual} != {expected}"
                )
            # Also validates that the prepared H/I/E dimensions match.
            SonicMoE(candidate, self.weights)

    def _candidate_fingerprint(self) -> list[dict[str, int | bool | None]]:
        return [_config_tuning_dict(config) for config in self.candidates]

    def _cache_key(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> str:
        props = torch.cuda.get_device_properties(hidden_states.device)
        identity = {
            "schema": _CACHE_SCHEMA_VERSION,
            "arch": str(get_rocm_arch()),
            "device": str(props.name),
            "compute_units": int(props.multi_processor_count),
            "flydsl": str(getattr(flydsl, "__version__", "")),
            "torch": str(torch.__version__),
            "torch_hip": str(torch.version.hip),
            "source": self._source_hash,
            "weight_dtype": self.weights.weight_dtype,
            "hidden_dtype": str(hidden_states.dtype),
            "router_dtype": str(router_logits.dtype),
            "tokens_bucket": _token_bucket(int(hidden_states.shape[0])),
            "hidden_size": self.base_config.hidden_size,
            "intermediate_size": self.base_config.intermediate_size,
            "num_experts": self.base_config.num_experts,
            "top_k": self.base_config.top_k,
            "renormalize": self.base_config.renormalize,
            "profile_key": self.profile_key,
            "validated": self.validate_candidates,
            "candidates": self._candidate_fingerprint(),
        }
        return json.dumps(identity, sort_keys=True, separators=(",", ":"))

    def _op_for(self, config: SonicMoEConfig) -> SonicMoE:
        op = self._ops.get(config)
        if op is None:
            op = SonicMoE(config, self.weights)
            self._ops[config] = op
        return op

    def _load_disk_cache(self) -> None:
        if not self.cache_file.is_file():
            return
        try:
            payload = json.loads(self.cache_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            if payload.get("version") != _CACHE_SCHEMA_VERSION:
                return
            entries = payload.get("entries", {})
            if not isinstance(entries, dict):
                return
            candidate_by_tuning = {
                json.dumps(_config_tuning_dict(config), sort_keys=True): config
                for config in self.candidates
            }
            for key, tuning in entries.items():
                encoded = json.dumps(tuning, sort_keys=True)
                candidate = candidate_by_tuning.get(encoded)
                if candidate is not None:
                    self._winner_cache[key] = candidate
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return

    def _save_disk_cache(self) -> None:
        if not self.validate_candidates:
            return
        temporary: Path | None = None
        try:
            import fcntl

            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.cache_file.with_name(f".{self.cache_file.name}.lock")
            with lock_path.open("a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                entries: dict[str, dict[str, int | bool | None]] = {}
                if self.cache_file.is_file():
                    try:
                        old = json.loads(self.cache_file.read_text(encoding="utf-8"))
                        if isinstance(old, dict) and old.get(
                            "version"
                        ) == _CACHE_SCHEMA_VERSION and isinstance(
                            old.get("entries"), dict
                        ):
                            entries.update(old["entries"])
                    except (OSError, TypeError, ValueError, json.JSONDecodeError):
                        pass
                entries.update(
                    {
                        key: _config_tuning_dict(config)
                        for key, config in self._winner_cache.items()
                    }
                )
                payload = {"version": _CACHE_SCHEMA_VERSION, "entries": entries}
                temporary = self.cache_file.with_name(
                    f".{self.cache_file.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
                )
                temporary.write_text(
                    json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
                )
                os.replace(temporary, self.cache_file)
                temporary = None
        except OSError:
            # Autotuning remains usable with its in-memory cache on read-only or
            # otherwise unavailable cache filesystems.
            return
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _candidate_matches(reference: torch.Tensor, candidate: torch.Tensor) -> bool:
        reference_f32 = reference.float()
        candidate_f32 = candidate.float()
        if not torch.isfinite(candidate_f32).all().item():
            return False
        ref_norm = torch.linalg.vector_norm(reference_f32)
        candidate_norm = torch.linalg.vector_norm(candidate_f32)
        if ref_norm.item() == 0.0 or candidate_norm.item() == 0.0:
            return torch.equal(reference, candidate)
        cosine = torch.nn.functional.cosine_similarity(
            reference_f32.flatten(), candidate_f32.flatten(), dim=0
        ).item()
        norm_ratio = (candidate_norm / ref_norm).item()
        max_reference = reference_f32.abs().max().item()
        max_error = (candidate_f32 - reference_f32).abs().max().item()
        return (
            cosine >= 0.999
            and 0.98 <= norm_ratio <= 1.02
            and max_error <= 0.15 * max(max_reference, 1.0e-2)
        )

    def _search(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> SonicMoEConfig:
        self.search_count += 1
        self.last_results = ()
        stream = torch.cuda.current_stream(hidden_states.device)
        scratch = torch.empty(
            (hidden_states.shape[0], self.base_config.hidden_size),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        reference: torch.Tensor | None = None
        if self.validate_candidates:
            base_op = self._op_for(self.base_config)
            try:
                base_op(hidden_states, router_logits, out=scratch)
                stream.synchronize()
                reference = scratch.clone()
            finally:
                base_op.clear_workspace()

        results: list[tuple[SonicMoEConfig, float]] = []
        for candidate in self.candidates:
            op: SonicMoE | None = None
            try:
                op = self._op_for(candidate)
                op(hidden_states, router_logits, out=scratch)
                stream.synchronize()
                if reference is not None and not self._candidate_matches(reference, scratch):
                    continue
                elapsed_ms = do_bench(
                    lambda: op(hidden_states, router_logits, out=scratch),
                    warmup=self.warmup,
                    rep=self.rep,
                )
                results.append((candidate, float(elapsed_ms)))
            except (AssertionError, RuntimeError, TypeError, ValueError):
                continue
            finally:
                if op is not None:
                    op.clear_workspace()
        if not results:
            self._ops.clear()
            raise RuntimeError("all SonicMoE autotune candidates failed")
        winner = min(results, key=lambda item: item[1])[0]
        winner_op = self._ops[winner]
        self._ops = {winner: winner_op}
        self.last_results = tuple(results)
        return winner

    def select(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        force: bool = False,
    ) -> SonicMoEConfig:
        key = self._cache_key(hidden_states, router_logits)
        winner = None if force else self._winner_cache.get(key)
        if winner is None:
            winner = self._search(hidden_states, router_logits)
            self._winner_cache[key] = winner
            self._save_disk_cache()
        self.best_config = winner
        return winner

    def __call__(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        force_tune: bool = False,
    ) -> torch.Tensor:
        winner = self.select(hidden_states, router_logits, force=force_tune)
        return self._op_for(winner)(hidden_states, router_logits, out=out)


__all__ = [
    "SonicMoEAutotuner",
    "default_sonic_moe_candidates",
]
