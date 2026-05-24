import torch

from forza_abyss_painter.shapegen.gpu.device import get_device, DTYPE


def test_get_device_returns_mps_on_apple_silicon():
    dev = get_device()
    if not torch.backends.mps.is_available():
        import pytest
        pytest.skip("MPS not available; this prototype targets Apple Silicon")
    assert dev.type == "mps"


def test_dtype_is_float32():
    # MPS support for float64 is patchy. Lock the prototype to float32.
    assert DTYPE == torch.float32
