import numpy as np
import pytest
from PIL import Image

from tools.create_line_guide_sobel_onnx import build_model


def test_reference_line_guide_onnx_generator_writes_expected_graph_markers():
    data = build_model()

    assert b"fd6_sobel_line_guide" in data
    assert b"rgb_to_luma" in data
    assert b"sobel_x" in data
    assert b"sobel_y" in data
    assert b"edge_strength" in data
    assert len(data) < 4096


def test_reference_line_guide_onnx_runs_when_runtime_is_available(tmp_path):
    pytest.importorskip("onnxruntime")
    from fd6.shapegen.line_guide import load_line_guide_onnx

    model_path = tmp_path / "line_guide.onnx"
    model_path.write_bytes(build_model())
    image = np.full((16, 16, 3), 220, dtype=np.uint8)
    image[4:12, 7:9] = 20

    guide, providers = load_line_guide_onnx(
        Image.fromarray(image, "RGB"),
        model_path,
        (16, 16),
        prefer_gpu=False,
    )

    assert "CPUExecutionProvider" in providers
    assert guide.shape == (16, 16)
    assert float(guide[:, 6:10].mean()) > float(guide[:, :3].mean())
