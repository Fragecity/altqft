from __future__ import annotations

import torch

SUPPORTED_TORCH_DEVICES = {"auto", "cpu", "cuda", "mps"}


def resolve_compute_device(device: str = "auto") -> str:
    normalized = device.lower()
    if normalized not in SUPPORTED_TORCH_DEVICES:
        supported = ", ".join(sorted(SUPPORTED_TORCH_DEVICES))
        raise ValueError(f"unsupported device '{device}', expected one of: {supported}")

    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"

    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda requested but no CUDA device is available")

    if normalized == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise ValueError("mps requested but no MPS device is available")

    return normalized


def available_cuda_device_count() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())
