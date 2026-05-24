import numpy as np
import torch

from forza_abyss_painter.shapegen.gpu.device import get_device, DTYPE
from forza_abyss_painter.shapegen.gpu.shapes_gpu import KINDS
from forza_abyss_painter.shapegen.shapes.rectangle import RotatedRectangle
from forza_abyss_painter.shapegen.shapes.triangle import Triangle


def _cpu_full_mask(shape, h, w):
    local, (x0, y0, x1, y1) = shape.rasterize_mask(w, h)
    full = np.zeros((h, w), dtype=np.float32)
    if local.size:
        full[y0:y1, x0:x1] = local.astype(np.float32) / 255.0
    return full


# ---- rotated_rectangle ----

def test_rect_hard_matches_cpu_rotated():
    h, w = 64, 64
    kind = KINDS["rotated_rectangle"]
    for angle in [0.0, 30.0, 45.0]:
        params = torch.tensor([[32.0, 32.0, 16.0, 8.0, angle]], dtype=DTYPE, device=get_device())
        gpu = kind.rasterize_hard(params, h, w).cpu().numpy()[0]
        cpu = _cpu_full_mask(RotatedRectangle(x=32, y=32, hw=16, hh=8, angle=angle), h, w)
        disagree = float(np.mean(np.abs(gpu - cpu)))
        assert disagree < 0.02, f"angle={angle}: rect hard disagrees {disagree}"


def test_rect_hard_area_matches_cpu():
    h, w = 96, 96
    kind = KINDS["rotated_rectangle"]
    params = torch.tensor([[48.0, 48.0, 20.0, 10.0, 0.0]], dtype=DTYPE, device=get_device())
    gpu_area = float(kind.rasterize_hard(params, h, w).sum().cpu())
    cpu_area = float(_cpu_full_mask(RotatedRectangle(x=48, y=48, hw=20, hh=10, angle=0.0), h, w).sum())
    assert abs(gpu_area - cpu_area) / cpu_area < 0.02, f"rect area gpu={gpu_area} cpu={cpu_area}"


def test_rect_soft_is_differentiable():
    h, w = 48, 48
    kind = KINDS["rotated_rectangle"]
    p = torch.tensor([[24.0, 24.0, 10.0, 6.0, 20.0]], dtype=DTYPE,
                     device=get_device(), requires_grad=True)
    kind.rasterize_soft(p, h, w).sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0, "soft rect produced zero gradient"


# ---- triangle ----

def test_tri_hard_matches_cpu():
    h, w = 64, 64
    kind = KINDS["triangle"]
    verts = [10.0, 10.0, 50.0, 15.0, 25.0, 55.0]
    params = torch.tensor([verts], dtype=DTYPE, device=get_device())
    gpu = kind.rasterize_hard(params, h, w).cpu().numpy()[0]
    cpu = _cpu_full_mask(Triangle(x1=verts[0], y1=verts[1], x2=verts[2], y2=verts[3],
                                  x3=verts[4], y3=verts[5]), h, w)
    disagree = float(np.mean(np.abs(gpu - cpu)))
    assert disagree < 0.02, f"triangle hard disagrees {disagree}"


def test_tri_hard_area_matches_cpu():
    h, w = 80, 80
    kind = KINDS["triangle"]
    verts = [10.0, 10.0, 50.0, 10.0, 10.0, 40.0]  # right triangle
    params = torch.tensor([verts], dtype=DTYPE, device=get_device())
    gpu_area = float(kind.rasterize_hard(params, h, w).sum().cpu())
    cpu_area = float(_cpu_full_mask(Triangle(x1=verts[0], y1=verts[1], x2=verts[2], y2=verts[3],
                                             x3=verts[4], y3=verts[5]), h, w).sum())
    assert abs(gpu_area - cpu_area) / cpu_area < 0.02, f"tri area gpu={gpu_area} cpu={cpu_area}"


def test_tri_hard_winding_independent():
    """CW and CCW vertex orders must produce the same filled triangle."""
    h, w = 64, 64
    kind = KINDS["triangle"]
    ccw = torch.tensor([[10.0, 10.0, 50.0, 15.0, 25.0, 55.0]], dtype=DTYPE, device=get_device())
    cw = torch.tensor([[10.0, 10.0, 25.0, 55.0, 50.0, 15.0]], dtype=DTYPE, device=get_device())
    m_ccw = kind.rasterize_hard(ccw, h, w).cpu().numpy()[0]
    m_cw = kind.rasterize_hard(cw, h, w).cpu().numpy()[0]
    assert float(np.mean(np.abs(m_ccw - m_cw))) < 0.01


def test_tri_soft_is_differentiable():
    h, w = 48, 48
    kind = KINDS["triangle"]
    p = torch.tensor([[8.0, 8.0, 38.0, 12.0, 20.0, 40.0]], dtype=DTYPE,
                     device=get_device(), requires_grad=True)
    kind.rasterize_soft(p, h, w).sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert p.grad.abs().sum() > 0, "soft triangle produced zero gradient"


# ---- registry sanity ----

def test_registry_init_and_json_roundtrip():
    for name, kind in KINDS.items():
        gen = torch.Generator(device="cpu").manual_seed(0)
        params = kind.init(4, 64, 64, gen)
        assert params.shape == (4, kind.param_count), f"{name} init wrong shape"
        d = kind.to_json(params[0].tolist(), [10, 20, 30, 128])
        assert d["type"] == name
        assert d["color"] == [10, 20, 30, 128]
