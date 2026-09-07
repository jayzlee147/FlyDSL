# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Correctness tests for SonicMoE's gfx950 grouped A16 NN kernel."""

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.common.tensor_shim import _run_compiled
from kernels.moe.sonic_grouped_a16w16 import compile_sonic_grouped_a16w16_nn

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


def _gfx950_device():
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU is required")
    arch = str(get_rocm_arch())
    if "gfx950" not in arch:
        pytest.skip(f"grouped A16 test requires gfx950, found {arch}")
    return torch.device("cuda")


@pytest.mark.parametrize(
    ("block_n", "n_waves"),
    ((64, 2), (128, 2), (256, 4)),
)
def test_compact_final_tile_writes_zero_dz_padding(block_n, n_waves):
    """A partial real-M tile is safe because backward produces zero padded dZ."""

    device = _gfx950_device()
    contraction_size, output_size, num_experts = 128, 256, 3
    rows = 128
    real_rows = 65
    generator = torch.Generator(device=device).manual_seed(443)

    # Expert 2 has 65 real rows, rounded to five BM16 descriptors.  Backward's
    # gather/activation chain makes the other 15 rows in the final tile zero.
    dz = torch.zeros((rows, contraction_size), dtype=torch.bfloat16, device=device)
    dz[:real_rows] = torch.randn(
        (real_rows, contraction_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    w1 = torch.randn(
        (num_experts, contraction_size, output_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    dx_sorted = torch.full(
        (rows, output_size),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    schedule = torch.tensor([5, 0, 1, 2, 3, 4], dtype=torch.int32, device=device)
    sorted_expert_ids = torch.tensor([2, 2], dtype=torch.int32, device=device)
    num_valid_ids = torch.tensor([rows, real_rows], dtype=torch.int32, device=device)

    grouped_dx = compile_sonic_grouped_a16w16_nn(
        contraction_size=contraction_size,
        output_size=output_size,
        num_experts=num_experts,
        block_m=16,
        block_n=block_n,
        block_k=64,
        stages=2,
        n_waves=n_waves,
        compact_grid=True,
        device_index=device.index or 0,
    )
    _run_compiled(
        grouped_dx,
        dz.data_ptr(),
        w1.data_ptr(),
        schedule.data_ptr(),
        sorted_expert_ids.data_ptr(),
        num_valid_ids.data_ptr(),
        dx_sorted.data_ptr(),
        3,  # Force multiple persistent grid-stride iterations.
        torch.cuda.current_stream(device),
    )
    torch.cuda.synchronize(device)

    expected = dz[:real_rows].float() @ w1[2].float()
    torch.testing.assert_close(
        dx_sorted[:real_rows].float(),
        expected,
        rtol=3e-2,
        atol=5e-2,
    )
    assert torch.count_nonzero(dx_sorted[real_rows:80]) == 0
    assert torch.isnan(dx_sorted[80:]).all()
