from __future__ import annotations

import torch

DTYPE = torch.float32


def get_device() -> torch.device:
    # CUDA first (Colab / discrete GPUs), then Apple MPS, then CPU. This lets the same
    # package code run on a Colab CUDA runtime and on an Apple Silicon Mac unchanged.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
