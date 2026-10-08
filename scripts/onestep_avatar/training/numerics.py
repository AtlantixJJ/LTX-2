"""One deterministic training policy, with no Torch import during queue planning.

Configure the exact cuBLAS workspace before native imports or CUDA work. Refuse
late missing/wrong workspace; backend detection may initialize CUDA after the
workspace is already set. Apply strict deterministic Torch/cuDNN algorithms,
disable cuDNN benchmarking and CUDA matmul TF32, then capture actual flags.
Preserve cuDNN's separate TF32 setting and require full original/serial agreement.
Inputs are inherited environment and optional explicit original observations;
outputs are checked actual policy records. Missing history never gains defaults.
"""

from __future__ import annotations

import os
import sys

ENVIRONMENT = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
POLICY = {"deterministic_algorithms": True, "deterministic_warn_only": False,
          "cudnn_deterministic": True, "cudnn_benchmark": False, "allow_tf32": False,
          "cublas_workspace": ":4096:8"}


def require_environment(record: dict) -> None:
    if record != ENVIRONMENT or any(os.environ.get(key) != value for key, value in record.items()):
        raise ValueError("numerical launch environment differs from the required inherited workspace")


def configure_environment(*, required: bool = False) -> None:
    """Bootstrap before native imports; never silently replace an inherited setting."""
    key, expected = next(iter(ENVIRONMENT.items()))
    actual = os.environ.get(key)
    if actual is None and not required:
        torch_module = sys.modules.get("torch")
        if torch_module is not None and torch_module.cuda.is_initialized():
            raise ValueError("cannot configure the numerical workspace after CUDA initialization")
        os.environ[key] = expected
    require_environment(ENVIRONMENT)


def capture() -> dict:
    import torch  # noqa: PLC0415 -- queue planning must remain import-light

    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def validate(record: dict, *, required: bool = False, expected: dict | None = None) -> None:
    fields = set(POLICY) | {"cudnn_allow_tf32"}
    if (not isinstance(record, dict) or set(record) != fields
            or any(type(record[key]) is not bool for key in fields - {"cublas_workspace"})
            or (record["cublas_workspace"] is not None and not isinstance(record["cublas_workspace"], str))):
        raise ValueError("observed numerical policy is malformed")
    if required and any(record[key] != value for key, value in POLICY.items()):
        raise ValueError("observed numerical policy differs from deterministic training")
    if expected is not None:
        validate(expected, required=True)
        if record != expected:
            raise ValueError("observed numerical policy differs from the original native reference")


def apply(*, expected: dict | None = None, environment_required: bool = False) -> dict:
    """Apply before model kernels; correctly inherited workspace permits backend detection."""
    if expected is not None:
        validate(expected, required=True)
    configure_environment(required=environment_required)
    import torch  # noqa: PLC0415 -- environment must be configured before Torch/native imports

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    observed = capture()
    validate(observed, required=True, expected=expected)
    return observed
