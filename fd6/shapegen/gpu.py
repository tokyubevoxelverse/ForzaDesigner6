"""Cross-vendor GPU acceleration (OpenCL via pyopencl) for the shape search.

Design goals (per the maintainer's brief):
  * **All GPUs** — OpenCL works on NVIDIA, AMD, and Intel through the user's
    graphics driver (no CUDA lock-in).
  * **Lean exe** — pyopencl is NOT bundled. It's downloaded + extracted on
    demand the first time the user explicitly selects GPU (see ensure_pyopencl);
    the heavy OpenCL runtime itself ships with the GPU driver.
  * **Never breaks the app** — if the runtime can't be installed, no GPU is
    present, or any GPU op raises, every entry point degrades to "use CPU".
  * **Output-stable** — the GPU only *ranks* candidate ellipses each iteration;
    the engine still commits the chosen shape with the CPU `composite()`, so the
    final colors/canvas match the CPU path.

`OpenCLEllipseSearcher` runs the live GPU path. `EllipseBatchSearcher` is an
array-module (NumPy/CuPy) reference kept for unit tests that prove numerical
parity with `scoring.score_shape`.

Scope: ellipses (`rotated_ellipse` / `ellipse`) only — the engine falls back to
the CPU path for any other shape type.
"""

from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from fd6.shapegen.shapes import Shape
from fd6.shapegen.shapes.ellipse import RotatedEllipse


# Alpha the search assumes for every candidate (matches RotatedEllipse.random /
# Ellipse.random, which fix color alpha at 128). The committed alpha is whatever
# the CPU composite picks; this constant only affects ranking.
_SEARCH_ALPHA = 128.0 / 255.0

# Memory guard: cap a single scoring mini-batch's tile tensor to this many cells
# (B * T * T). Keeps peak VRAM bounded regardless of canvas size / sample count.
_MAX_TILE_CELLS = 48_000_000

_GPU_PROBE_CACHE: Optional[bool] = None
_RUNTIME_READY: Optional[bool] = None
_GPU_NAMES: list[str] = []
_INSTALL_LOG: str = ""

def _runtime_dir() -> Path:
    """User-writable dir the GPU runtime is installed into on demand."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return Path(base) / "FD6" / "gpu_runtime"


def _pip_install_target(target: Path, packages: list[str]) -> bool:
    """Install `packages` (+ their deps) into `target` using pip in-process.

    pip resolves the correct wheels for THIS interpreter — ABI, platform, and
    the full dependency tree (pyopencl → pytools, siphash24, platformdirs) — so
    we don't hand-pick wheels (which got the ABI variant wrong). pip is bundled
    with the frozen app and already present on a source checkout. The frozen exe
    can't shell out to `python -m pip` (sys.executable is the app), so we call
    pip's entry point in-process.
    """
    global _INSTALL_LOG
    try:
        from pip._internal.cli.main import main as pip_main
    except Exception as exc:
        _INSTALL_LOG = f"pip unavailable: {type(exc).__name__}: {exc}"
        return False
    args = [
        "install", "--target", str(target), "--no-input",
        "--disable-pip-version-check", "--no-warn-script-location",
        *packages,
    ]
    try:
        rc = pip_main(args)
    except SystemExit as e:
        rc = int(getattr(e, "code", 1) or 0)
    except Exception as exc:
        _INSTALL_LOG = f"pip error: {type(exc).__name__}: {exc}"
        return False
    if rc != 0:
        _INSTALL_LOG = f"pip exited with code {rc}"
    return rc == 0


def ensure_pyopencl(install: bool = True) -> bool:
    """Make `pyopencl` importable, installing it on demand if missing.

    Order: (1) already importable? (2) already installed in the runtime dir?
    (3) pip-install it there now (if install=True). Cached. Any failure returns
    False so the caller falls back to CPU — GPU is best-effort, never blocks the
    app. The runtime dir is appended to sys.path (not prepended) so the app's
    bundled numpy keeps priority over pip's copy.
    """
    global _RUNTIME_READY, _INSTALL_LOG
    if _RUNTIME_READY:
        return True
    try:
        import pyopencl  # noqa: F401
        _RUNTIME_READY = True
        return True
    except Exception:
        pass
    d = _runtime_dir()
    if d.exists() and str(d) not in sys.path:
        sys.path.append(str(d))
        try:
            import importlib
            importlib.invalidate_caches()
            import pyopencl  # noqa: F401
            _RUNTIME_READY = True
            return True
        except Exception:
            pass
    if not install:
        return False
    try:
        d.mkdir(parents=True, exist_ok=True)
        if not _pip_install_target(d, ["pyopencl"]):
            return False
        if str(d) not in sys.path:
            sys.path.append(str(d))
        import importlib
        importlib.invalidate_caches()
        import pyopencl  # noqa: F401
        _RUNTIME_READY = True
        return True
    except Exception as exc:
        _INSTALL_LOG = f"{type(exc).__name__}: {exc}"
        return False


def install_log() -> str:
    return _INSTALL_LOG


def _enumerate_gpu_names() -> list[str]:
    import pyopencl as cl
    names: list[str] = []
    for plat in cl.get_platforms():
        try:
            for dev in plat.get_devices():
                if int(dev.type) & int(cl.device_type.GPU):
                    names.append(dev.name.strip())
        except Exception:
            continue
    return names


def gpu_available(install: bool = True) -> bool:
    """True iff a usable OpenCL GPU device is present (cross-vendor: NVIDIA /
    AMD / Intel). Installs the pyopencl runtime on demand. Cached."""
    global _GPU_PROBE_CACHE, _GPU_NAMES
    if _GPU_PROBE_CACHE is not None:
        return _GPU_PROBE_CACHE
    ok = False
    try:
        if ensure_pyopencl(install=install):
            _GPU_NAMES = _enumerate_gpu_names()
            ok = len(_GPU_NAMES) > 0
    except Exception:
        ok = False
    _GPU_PROBE_CACHE = ok
    return ok


def gpu_detected_without_install() -> bool:
    """Light probe used by the UI: is pyopencl ALREADY present + a GPU visible,
    WITHOUT triggering a download? Avoids a network install at startup."""
    try:
        return ensure_pyopencl(install=False) and len(_enumerate_gpu_names()) > 0
    except Exception:
        return False


def list_gpu_names() -> list[str]:
    gpu_available()
    return list(_GPU_NAMES)


def resolve_backend(requested: str) -> str:
    """Map a profile's compute_backend ('auto'|'cpu'|'gpu') to 'cpu' or 'gpu'.

    'auto' -> 'gpu' when an OpenCL GPU is available else 'cpu'. 'gpu' -> 'gpu'
    only when actually available (otherwise 'cpu', so a saved profile can't
    wedge the app on a machine without a GPU). 'cpu' is always honored.
    """
    req = (requested or "auto").lower().strip()
    if req == "cpu":
        return "cpu"
    if req == "gpu":
        # Explicit GPU request → use the bundled pyopencl if present, else try a
        # one-time on-demand install as a fallback (source/dev runs).
        return "gpu" if gpu_available(install=True) else "cpu"
    if req == "auto":
        # pyopencl is bundled in the exe, so "already importable" is the common
        # case — Auto uses the GPU whenever a device is visible WITHOUT ever
        # triggering a network install.
        return "gpu" if gpu_detected_without_install() else "cpu"
    return "cpu"


def backend_label(backend: str) -> str:
    if backend != "gpu":
        return "CPU"
    names = list_gpu_names()
    return f"GPU ({names[0]})" if names else "GPU (OpenCL)"


def _xp():
    """Return the cupy module (raises if unavailable). Retained so
    EllipseBatchSearcher can still run on CuPy for tests/parity; the live engine
    uses the OpenCL searcher below."""
    import cupy as cp  # type: ignore
    return cp


# ── shared candidate-parameter generation (host side) ────────────────────────

def random_ellipse_params(w: int, h: int, b: int, max_size_frac: Optional[float],
                          rng: random.Random) -> np.ndarray:
    """B x 5 float32 (cx, cy, rx, ry, angle_deg). Matches RotatedEllipse.random."""
    if max_size_frac is None:
        rx_cap = max(2.0, w / 8.0)
        ry_cap = max(2.0, h / 8.0)
    else:
        rx_cap = max(2.0, (w * max_size_frac) / 2.0)
        ry_cap = max(2.0, (h * max_size_frac) / 2.0)
    rs = np.random.RandomState(rng.randint(0, 2**31 - 1))
    out = np.empty((b, 5), dtype=np.float32)
    out[:, 0] = rs.uniform(0, w - 1, b)
    out[:, 1] = rs.uniform(0, h - 1, b)
    out[:, 2] = rs.uniform(1, rx_cap, b)
    out[:, 3] = rs.uniform(1, ry_cap, b)
    out[:, 4] = rs.uniform(0, 180, b)
    return out


def mutate_ellipse_params(base: np.ndarray, w: int, h: int, b: int,
                          rng: random.Random) -> np.ndarray:
    """B jittered copies of `base` (mirrors RotatedEllipse.mutate distributions)."""
    rs = np.random.RandomState(rng.randint(0, 2**31 - 1))
    out = np.tile(base.astype(np.float32), (b, 1))
    kind = rs.randint(0, 4, b)
    m0 = kind == 0
    out[m0, 0] = np.clip(out[m0, 0] + rs.normal(0, 16, int(m0.sum())), 0, w - 1)
    out[m0, 1] = np.clip(out[m0, 1] + rs.normal(0, 16, int(m0.sum())), 0, h - 1)
    m1 = kind == 1
    out[m1, 2] = np.clip(out[m1, 2] + rs.normal(0, 16, int(m1.sum())), 1, w)
    out[m1, 3] = np.clip(out[m1, 3] + rs.normal(0, 16, int(m1.sum())), 1, h)
    m2 = kind == 2
    out[m2, 4] = np.mod(out[m2, 4] + rs.normal(0, 25, int(m2.sum())), 180.0)
    m3 = kind == 3
    out[m3, 0] = np.clip(out[m3, 0] + rs.normal(0, 8, int(m3.sum())), 0, w - 1)
    out[m3, 1] = np.clip(out[m3, 1] + rs.normal(0, 8, int(m3.sum())), 0, h - 1)
    out[m3, 4] = np.mod(out[m3, 4] + rs.normal(0, 15, int(m3.sum())), 180.0)
    return out


# ── OpenCL kernel: one work-item scores one candidate ellipse over its tile ──
_OPENCL_KERNEL = r"""
__kernel void score_ellipses(
    __global const float* canvas,   // H*W*3
    __global const float* target,   // H*W*3
    __global const float* edge,     // H*W
    __global const float* alpha,    // H*W (used only when has_alpha)
    __global const float* params,   // B*5: cx,cy,rx,ry,angle_deg
    __global float* out_scores,     // B
    const int W, const int H, const int T, const int B,
    const float full_sq, const float ninv, const float a, const int has_alpha)
{
    int b = get_global_id(0);
    if (b >= B) return;
    float cx = params[b*5+0];
    float cy = params[b*5+1];
    float rx = fmax(params[b*5+2], 1e-6f);
    float ry = fmax(params[b*5+3], 1e-6f);
    float ang = params[b*5+4] * 0.01745329252f;
    float ca = cos(ang), sa = sin(ang);
    int x0 = (int)round(cx) - T/2;
    int y0 = (int)round(cy) - T/2;

    // pass 1: optimal color over the masked (∩ alpha) region
    float eff_sum = 0.0f, n0 = 0.0f, n1 = 0.0f, n2 = 0.0f;
    for (int dy = 0; dy < T; ++dy) {
        int gy = y0 + dy; if (gy < 0 || gy >= H) continue;
        for (int dx = 0; dx < T; ++dx) {
            int gx = x0 + dx; if (gx < 0 || gx >= W) continue;
            float fx = (float)gx - cx, fy = (float)gy - cy;
            float xr = ca*fx + sa*fy, yr = -sa*fx + ca*fy;
            if ((xr/rx)*(xr/rx) + (yr/ry)*(yr/ry) > 1.0f) continue;
            int idx = gy*W + gx;
            float al = has_alpha ? alpha[idx] : 255.0f;
            float eff = al / 255.0f;
            if (eff <= 0.0f) continue;
            int c3 = idx*3;
            eff_sum += eff;
            n0 += eff*(target[c3]   - (1.0f-a)*canvas[c3]);
            n1 += eff*(target[c3+1] - (1.0f-a)*canvas[c3+1]);
            n2 += eff*(target[c3+2] - (1.0f-a)*canvas[c3+2]);
        }
    }
    float colR = 0.0f, colG = 0.0f, colB = 0.0f;
    if (eff_sum > 0.5f) {
        colR = floor(clamp(n0/(eff_sum*a), 0.0f, 255.0f));
        colG = floor(clamp(n1/(eff_sum*a), 0.0f, 255.0f));
        colB = floor(clamp(n2/(eff_sum*a), 0.0f, 255.0f));
    }

    // pass 2: edge-weighted error delta + sticker overlap, over mask==1 pixels
    float region_old = 0.0f, region_new = 0.0f, body = 0.0f, opaque = 0.0f;
    for (int dy = 0; dy < T; ++dy) {
        int gy = y0 + dy; if (gy < 0 || gy >= H) continue;
        for (int dx = 0; dx < T; ++dx) {
            int gx = x0 + dx; if (gx < 0 || gx >= W) continue;
            float fx = (float)gx - cx, fy = (float)gy - cy;
            float xr = ca*fx + sa*fy, yr = -sa*fx + ca*fy;
            if ((xr/rx)*(xr/rx) + (yr/ry)*(yr/ry) > 1.0f) continue;
            int idx = gy*W + gx;
            body += 1.0f;
            if (has_alpha && alpha[idx] >= 128.0f) opaque += 1.0f;
            float wgt = edge[idx];
            int c3 = idx*3;
            float curR=canvas[c3], curG=canvas[c3+1], curB=canvas[c3+2];
            float tgR=target[c3], tgG=target[c3+1], tgB=target[c3+2];
            float bR = a*colR + (1.0f-a)*curR;
            float bG = a*colG + (1.0f-a)*curG;
            float bB = a*colB + (1.0f-a)*curB;
            float oR=curR-tgR, oG=curG-tgG, oB=curB-tgB;
            float nR=bR-tgR, nG=bG-tgG, nB=bB-tgB;
            region_old += wgt*(oR*oR+oG*oG+oB*oB);
            region_new += wgt*(nR*nR+nG*nG+nB*nB);
        }
    }
    float total = full_sq - region_old + region_new;
    float score = (total > 0.0f) ? sqrt(total*ninv) : 0.0f;
    if (has_alpha) {
        if (body < 1.0f || (opaque / fmax(body, 1.0f)) < 0.995f) score = INFINITY;
    }
    out_scores[b] = score;
}
"""


class OpenCLEllipseSearcher:
    """Cross-vendor GPU ellipse search via pyopencl. Same scoring math as
    `scoring.score_shape` (edge-weighted RMS delta + sticker rejection); one
    OpenCL work-item scores one candidate over its tile. Only RANKS candidates —
    the engine commits the winner with the CPU `composite()`, so output matches
    the CPU path. Any failure raises so the engine falls back to CPU.
    """

    def __init__(self, target: np.ndarray, alpha_mask: Optional[np.ndarray],
                 edge_weight: np.ndarray) -> None:
        import pyopencl as cl
        self.cl = cl
        self.h, self.w = target.shape[:2]
        self._target_np = np.ascontiguousarray(target, dtype=np.float32)
        self._edge_np = np.ascontiguousarray(edge_weight, dtype=np.float32)
        self._has_alpha = alpha_mask is not None
        alpha_np = (np.ascontiguousarray(alpha_mask, dtype=np.float32)
                    if alpha_mask is not None else np.zeros((self.h, self.w), np.float32))
        self.n = float(self._edge_np.sum()) * 3.0
        self._ninv = (1.0 / self.n) if self.n >= 1.0 else 0.0

        dev = None
        for plat in cl.get_platforms():
            gpus = [d for d in plat.get_devices() if int(d.type) & int(cl.device_type.GPU)]
            if gpus:
                dev = gpus[0]
                break
        self.ctx = cl.Context(devices=[dev]) if dev is not None else cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)
        self.prog = cl.Program(self.ctx, _OPENCL_KERNEL).build()
        # Retrieve the kernel ONCE and reuse it — re-fetching prog.score_ellipses
        # per batch builds a fresh kernel object each time (pyopencl warns this is
        # "considerable expense").
        self.kernel = cl.Kernel(self.prog, "score_ellipses")

        mf = cl.mem_flags
        self._buf_target = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self._target_np.ravel())
        self._buf_edge = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self._edge_np.ravel())
        self._buf_alpha = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=alpha_np.ravel())
        self._buf_canvas = cl.Buffer(self.ctx, mf.READ_ONLY, size=self._target_np.nbytes)

    @staticmethod
    def _tile_for(params: np.ndarray, w: int, h: int) -> int:
        max_r = float(np.max(np.maximum(params[:, 2], params[:, 3]))) if params.size else 1.0
        return max(2, int(min(max(w, h), 2 * math.ceil(max_r) + 2)))

    def _score(self, params: np.ndarray, full_sq: float, T: int) -> np.ndarray:
        cl = self.cl
        mf = cl.mem_flags
        B = params.shape[0]
        p = np.ascontiguousarray(params, dtype=np.float32).ravel()
        buf_p = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=p)
        out = np.empty(B, dtype=np.float32)
        buf_out = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=out.nbytes)
        self.kernel(
            self.queue, (B,), None,
            self._buf_canvas, self._buf_target, self._buf_edge, self._buf_alpha,
            buf_p, buf_out,
            np.int32(self.w), np.int32(self.h), np.int32(T), np.int32(B),
            np.float32(full_sq), np.float32(self._ninv), np.float32(_SEARCH_ALPHA),
            np.int32(1 if self._has_alpha else 0),
        )
        cl.enqueue_copy(self.queue, out, buf_out)
        self.queue.finish()
        buf_p.release()
        buf_out.release()
        return out

    def search(self, canvas: np.ndarray, n_random: int, n_mutate: int,
               max_size_frac: Optional[float], rng: random.Random) -> tuple[float, Optional[Shape]]:
        cl = self.cl
        cur = np.ascontiguousarray(canvas, dtype=np.float32)
        cl.enqueue_copy(self.queue, self._buf_canvas, cur.ravel())
        full_sq = float((((cur - self._target_np) ** 2) * self._edge_np[:, :, None]).sum())

        params = random_ellipse_params(self.w, self.h, max(1, n_random), max_size_frac, rng)
        scores = self._score(params, full_sq, self._tile_for(params, self.w, self.h))
        bi = int(np.argmin(scores))
        best_score = float(scores[bi])
        best = params[bi].copy()
        if not math.isfinite(best_score):
            return float("inf"), None

        cap = max(1, n_mutate)
        batch = min(cap, 64)
        steps = max(1, cap // batch)
        no_improve = 0
        for _ in range(steps):
            muts = mutate_ellipse_params(best, self.w, self.h, batch, rng)
            mscores = self._score(muts, full_sq, self._tile_for(muts, self.w, self.h))
            mbi = int(np.argmin(mscores))
            ms = float(mscores[mbi])
            if ms < best_score:
                best_score, best = ms, muts[mbi].copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= max(2, steps // 4):
                    break

        cx, cy, rx, ry, ang = (float(v) for v in best)
        return best_score, RotatedEllipse(color=(0, 0, 0, 128), x=cx, y=cy, rx=rx, ry=ry, angle=ang)


class EllipseBatchSearcher:
    """Batched random-search + hill-climb for ellipses on a chosen array module.

    Mirrors `scoring.score_shape`'s edge-weighted formula
    (`total = full_sq - region_old + region_new`, normalized by the edge-weight
    sum) and its sticker overlap-rejection, but evaluates a whole batch of
    candidates at once over per-candidate tiles.
    """

    def __init__(self, target: np.ndarray, alpha_mask: Optional[np.ndarray],
                 edge_weight: np.ndarray, xp=None) -> None:
        self.xp = xp if xp is not None else _xp()
        xp = self.xp
        self.h, self.w = target.shape[:2]
        self.target = xp.asarray(target, dtype=xp.float32)
        self.edge = xp.asarray(edge_weight, dtype=xp.float32)
        self.alpha = None if alpha_mask is None else xp.asarray(alpha_mask, dtype=xp.float32)
        # Normalizer n = (sum of edge weights) * 3 channels — matches
        # precompute_canvas_error's edge-weighted branch.
        self.n = float(self.edge.sum().item()) * 3.0

    # ── public ────────────────────────────────────────────────────────────
    def search(self, canvas: np.ndarray, n_random: int, n_mutate: int,
               max_size_frac: Optional[float], rng: random.Random) -> tuple[float, Optional[Shape]]:
        """Return (best_score, best_shape) for one iteration. Shape may be None."""
        xp = self.xp
        cur = xp.asarray(canvas, dtype=xp.float32)
        # Full-canvas weighted squared error (constant for this canvas snapshot).
        full_sq = float(((cur - self.target) ** 2 * self.edge[:, :, None]).sum().item())

        # ── random search ──
        params = self._random_params(max(1, n_random), max_size_frac, rng)
        scores, _colors = self._score_batch(params, cur, full_sq)
        bi = int(xp.argmin(scores).item())
        best_score = float(scores[bi].item())
        best = params[bi].copy()
        if not math.isfinite(best_score):
            return float("inf"), None

        # ── hill climb (batched mutations) ──
        cap = max(1, n_mutate)
        batch = min(cap, 64)
        no_improve = 0
        steps = max(1, cap // batch)
        for _ in range(steps):
            muts = self._mutate_params(best, batch, rng)
            mscores, _mc = self._score_batch(muts, cur, full_sq)
            mbi = int(xp.argmin(mscores).item())
            ms = float(mscores[mbi].item())
            if ms < best_score:
                best_score, best = ms, muts[mbi].copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= max(2, steps // 4):
                    break

        cx, cy, rx, ry, ang = (float(v) for v in best)
        shape = RotatedEllipse(color=(0, 0, 0, 128), x=cx, y=cy, rx=rx, ry=ry, angle=ang)
        return best_score, shape

    # ── internals ─────────────────────────────────────────────────────────
    def _random_params(self, b: int, max_size_frac: Optional[float], rng: random.Random) -> np.ndarray:
        """B x 5 float32 (cx, cy, rx, ry, angle_deg) on host (cheap to build)."""
        w, h = self.w, self.h
        if max_size_frac is None:
            rx_cap = max(2.0, w / 8.0)
            ry_cap = max(2.0, h / 8.0)
        else:
            rx_cap = max(2.0, (w * max_size_frac) / 2.0)
            ry_cap = max(2.0, (h * max_size_frac) / 2.0)
        rs = np.random.RandomState(rng.randint(0, 2**31 - 1))
        out = np.empty((b, 5), dtype=np.float32)
        out[:, 0] = rs.uniform(0, w - 1, b)
        out[:, 1] = rs.uniform(0, h - 1, b)
        out[:, 2] = rs.uniform(1, rx_cap, b)
        out[:, 3] = rs.uniform(1, ry_cap, b)
        out[:, 4] = rs.uniform(0, 180, b)
        return out

    def _mutate_params(self, base: np.ndarray, b: int, rng: random.Random) -> np.ndarray:
        """B jittered copies of `base` (mirrors RotatedEllipse.mutate distributions)."""
        w, h = self.w, self.h
        rs = np.random.RandomState(rng.randint(0, 2**31 - 1))
        out = np.tile(base.astype(np.float32), (b, 1))
        kind = rs.randint(0, 4, b)
        # position jitter
        m0 = kind == 0
        out[m0, 0] = np.clip(out[m0, 0] + rs.normal(0, 16, m0.sum()), 0, w - 1)
        out[m0, 1] = np.clip(out[m0, 1] + rs.normal(0, 16, m0.sum()), 0, h - 1)
        # radius jitter
        m1 = kind == 1
        out[m1, 2] = np.clip(out[m1, 2] + rs.normal(0, 16, m1.sum()), 1, w)
        out[m1, 3] = np.clip(out[m1, 3] + rs.normal(0, 16, m1.sum()), 1, h)
        # angle jitter
        m2 = kind == 2
        out[m2, 4] = np.mod(out[m2, 4] + rs.normal(0, 25, m2.sum()), 180.0)
        # small combined jitter
        m3 = kind == 3
        out[m3, 0] = np.clip(out[m3, 0] + rs.normal(0, 8, m3.sum()), 0, w - 1)
        out[m3, 1] = np.clip(out[m3, 1] + rs.normal(0, 8, m3.sum()), 0, h - 1)
        out[m3, 4] = np.mod(out[m3, 4] + rs.normal(0, 15, m3.sum()), 180.0)
        return out

    def _score_batch(self, params: np.ndarray, cur, full_sq: float):
        """Score a B x 5 param array. Returns (scores[B], colors[B,3]) on `xp`.

        Mini-batches over candidates so the (B, T, T) tile tensors stay within
        the VRAM cap. T is sized to the largest ellipse in the batch.
        """
        xp = self.xp
        b_total = params.shape[0]
        # Tile side covering the biggest candidate (centered tiles, half = radius).
        max_r = float(np.max(np.maximum(params[:, 2], params[:, 3]))) if b_total else 1.0
        T = int(min(max(self.w, self.h), 2 * math.ceil(max_r) + 2))
        T = max(2, T)
        per = max(1, int(_MAX_TILE_CELLS // (T * T)))
        scores = xp.empty(b_total, dtype=xp.float32)
        colors = xp.zeros((b_total, 3), dtype=xp.float32)
        for start in range(0, b_total, per):
            sl = slice(start, min(b_total, start + per))
            s, c = self._score_chunk(xp.asarray(params[sl], dtype=xp.float32), cur, full_sq, T)
            scores[sl] = s
            colors[sl] = c
        return scores, colors

    def _score_chunk(self, p, cur, full_sq: float, T: int):
        xp = self.xp
        B = p.shape[0]
        cx = p[:, 0][:, None, None]
        cy = p[:, 1][:, None, None]
        rx = xp.maximum(p[:, 2], 1e-6)[:, None, None]
        ry = xp.maximum(p[:, 3], 1e-6)[:, None, None]
        ang = xp.deg2rad(p[:, 4])[:, None, None]
        cos_a = xp.cos(ang)
        sin_a = xp.sin(ang)

        # Centered integer tile: top-left = round(center) - T//2.
        x0 = xp.round(p[:, 0]).astype(xp.int64) - T // 2  # (B,)
        y0 = xp.round(p[:, 1]).astype(xp.int64) - T // 2
        lx = xp.arange(T)
        gx = x0[:, None, None] + lx[None, None, :]   # (B,1,T) -> broadcast
        gy = y0[:, None, None] + lx[None, :, None]   # (B,T,1)
        gx = xp.broadcast_to(gx, (B, T, T))
        gy = xp.broadcast_to(gy, (B, T, T))

        in_x = (gx >= 0) & (gx < self.w)
        in_y = (gy >= 0) & (gy < self.h)
        valid = (in_x & in_y).astype(xp.float32)        # (B,T,T)
        gxc = xp.clip(gx, 0, self.w - 1)
        gyc = xp.clip(gy, 0, self.h - 1)

        cur_t = cur[gyc, gxc]                            # (B,T,T,3)
        tgt_t = self.target[gyc, gxc]
        edge_t = self.edge[gyc, gxc] * valid             # (B,T,T) zeroed out-of-bounds

        # Rotated-ellipse mask (binary, matches RotatedEllipse.rasterize_mask).
        xrel = gx.astype(xp.float32) - cx
        yrel = gy.astype(xp.float32) - cy
        xr = cos_a * xrel + sin_a * yrel
        yr = -sin_a * xrel + cos_a * yrel
        inside = ((xr / rx) ** 2 + (yr / ry) ** 2) <= 1.0
        mask = (inside.astype(xp.float32)) * valid       # (B,T,T)

        # Effective color mask = mask ∩ alpha (sticker-safe optimal color).
        if self.alpha is not None:
            alpha_t = self.alpha[gyc, gxc] * valid
            eff = mask * (alpha_t / 255.0)
        else:
            alpha_t = None
            eff = mask

        a = _SEARCH_ALPHA
        # Closed-form optimal color over the effective-masked region. The
        # weight<0.5 guard matches compute_optimal_color exactly.
        eff_sum = eff.sum(axis=(1, 2))                   # (B,)
        denom = eff_sum * a
        numer = (eff[..., None] * (tgt_t - (1.0 - a) * cur_t)).sum(axis=(1, 2))  # (B,3)
        safe = eff_sum > 0.5
        d = xp.where(safe, denom, xp.float32(1.0))[:, None]
        color = xp.where(safe[:, None], xp.clip(numer / d, 0, 255), 0.0)
        color = xp.floor(color)  # match compute_optimal_color's int32 truncation

        # Blended tile with that color, then edge-weighted region delta.
        m = mask[..., None]
        blended = m * (a * color[:, None, None, :] + (1.0 - a) * cur_t) + (1.0 - m) * cur_t
        w_t = edge_t[..., None]
        region_old = (w_t * (cur_t - tgt_t) ** 2).sum(axis=(1, 2, 3))   # (B,)
        region_new = (w_t * (blended - tgt_t) ** 2).sum(axis=(1, 2, 3))
        total = full_sq - region_old + region_new
        n = self.n if self.n >= 1.0 else 1.0
        score = xp.sqrt(xp.maximum(total, 0.0) / n)

        # Sticker overlap rejection (matches STICKER_OVERLAP_MIN=0.995) when an
        # alpha mask is present (it always is — edge-buffer ring or silhouette).
        if alpha_t is not None:
            body = (mask >= 0.5).astype(xp.float32)
            body_total = body.sum(axis=(1, 2))
            opaque = ((alpha_t >= 128.0) & (mask >= 0.5)).astype(xp.float32).sum(axis=(1, 2))
            ratio = xp.where(body_total >= 1.0, opaque / xp.maximum(body_total, 1.0), 0.0)
            reject = (body_total < 1.0) | (ratio < 0.995)
            score = xp.where(reject, xp.float32(np.inf), score)
        return score, color
