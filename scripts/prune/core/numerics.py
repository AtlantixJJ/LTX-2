"""Deterministic numerical policy for every native pruning model process.

Call `deterministic_policy()` before the first CUDA kernel. It pins the cuBLAS
workspace to `:4096:8`, enables deterministic algorithms, disables cuDNN
benchmarking and TF32 matmul. A different workspace after CUDA initialization
cannot take effect, so the function fails rather than silently continuing.
"""

import os

import torch

CUBLAS_WORKSPACE = ":4096:8"


def deterministic_policy():
    if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE:
        raise ValueError("cuBLAS policy must be set before CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
