import os
import random

import numpy as np
import pytest
import sgl_kernel_npu
import torch
import torch_npu

from sgl_kernel_npu.fla.solve_tril import solve_tril_npu as solve_tril
from sgl_kernel_npu.fla.chunk import fast_inv_tril_wrapper

# @pytest.mark.parametrize("name", [f"tensors/data_{n}.pt" for n in range(1, 828)])
@pytest.mark.parametrize("name", [f"tensors/data_40.pt"])
def test(name):
    attn_trill = torch.load(name)  # torch.Tensor, [1, 6, 4, 64]

    out_ref = solve_tril(attn_trill)
    out = fast_inv_tril_wrapper(attn_trill)
    # print(out)
    # print(out_ref)
    x = out - out_ref
    with open("tensor.txt", "w") as f:
        for n in range(x.shape[0]):
            f.write(f"Batch {n}\n")
            for c in range(x.shape[1]):
                f.write(f" Channel {c}\n")
                f.write(str(x[n, c].cpu().numpy()) + "\n\n")

    assert torch.allclose(
        out, out_ref.to(torch.float32), atol=1e-3
    ), f"Max difference is {(out - out_ref).abs().max()}, {attn_trill.shape}{out.shape}{out_ref.shape}"
    print("OK!")
