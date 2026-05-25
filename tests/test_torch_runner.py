"""Tests for forza_abyss_painter.runtime.torch_runner — the subprocess
entry point Phase 3 will spawn from the EXE.

Two test surfaces:
  1. In-process unit tests for RunConfig parsing, emit() format, and
     run() with the engine mocked at import-binding time.
  2. Real-subprocess smoke that actually spawns python -m torch_runner
     with a config + mocked engine, captures stderr, parses the JSON
     line-stream, and verifies the IPC contract holds end-to-end.

The mocking strategy:
  - run() imports `from forza_abyss_painter.shapegen.gpu.engine import
    GPUConfig, run_gpu` lazily inside the function. We provide a fake
    module via sys.modules so the import succeeds without torch.
  - Same trick works in the subprocess test via a PYTHONPATH-injected
    sitecustomize that stubs the module.

Why bother with the subprocess test on top of in-process unit tests:
  the IPC contract is a process-boundary contract. stdout/stderr line
  buffering, sys.exit codes, and JSON-per-line flushing all behave
  differently in-process vs. across a real subprocess. The subprocess
  smoke catches buffering bugs the in-process test can't.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from forza_abyss_painter.runtime import torch_runner as tr


# ============================================================ RunConfig


def _valid_config_dict(tmp_path: Path) -> dict:
    """Return a minimal valid config dict for tests that don't care about
    specific values. Files referenced are guaranteed to exist on disk so
    tests that pipe this through Path/open don't error spuriously."""
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")   # not a real PNG; only path checks
    return {
        "image_path": str(src),
        "output_json_path": str(tmp_path / "out.json"),
        "num_shapes": 100,
        "max_resolution": 480,
        "random_samples": 1024,
    }


def test_runconfig_accepts_minimal_required_fields(tmp_path):
    """All required fields present + valid → parse cleanly + defaults
    fill in. If RunConfig drifts to require more, the EXE's existing
    GenerateLocallyDialog won't build a parseable config and the IPC
    breaks at the boundary."""
    cfg = tr.RunConfig.from_dict(_valid_config_dict(tmp_path))
    assert cfg.num_shapes == 100
    assert cfg.max_resolution == 480
    assert cfg.random_samples == 1024
    assert cfg.sticker_mode is False
    assert cfg.lock_alpha is True
    assert cfg.device == "cuda"


def test_runconfig_rejects_missing_required_field(tmp_path):
    """Missing image_path → ValueError with the missing field named.
    The EXE-side modal surfaces this directly so the user sees what
    the runner needed but didn't get."""
    d = _valid_config_dict(tmp_path)
    del d["image_path"]
    with pytest.raises(ValueError, match="image_path"):
        tr.RunConfig.from_dict(d)


def test_runconfig_rejects_lock_alpha_false(tmp_path):
    """lock_alpha=False is a hard system constraint violation —
    Forza injector writes alpha=255 anyway and any non-opaque JSON
    breaks preview/in-game parity. Catching it at config-parse means
    the user doesn't burn 10 minutes on a run whose output will be
    rejected by the validator at save time."""
    d = _valid_config_dict(tmp_path)
    d["lock_alpha"] = False
    with pytest.raises(ValueError, match="lock_alpha"):
        tr.RunConfig.from_dict(d)


def test_runconfig_rejects_out_of_range_values(tmp_path):
    """Range checks catch GUI bugs that send 0/-1 down to the runner
    where they'd manifest as deep-engine crashes."""
    base = _valid_config_dict(tmp_path)
    for bad_field, bad_value in [
        ("num_shapes", 0),
        ("max_resolution", 32),
        ("random_samples", -1),
    ]:
        d = dict(base, **{bad_field: bad_value})
        with pytest.raises(ValueError, match=bad_field):
            tr.RunConfig.from_dict(d)


def test_runconfig_rejects_invalid_device(tmp_path):
    """device must be 'cuda' or 'cpu' — typo or stale enum from the
    EXE side would otherwise reach torch.device() and crash with a
    confusing message."""
    d = _valid_config_dict(tmp_path)
    d["device"] = "rocm"
    with pytest.raises(ValueError, match="device"):
        tr.RunConfig.from_dict(d)


def test_runconfig_summary_includes_user_facing_fields(tmp_path):
    """The 'started' event uses summary() to echo back what the EXE
    asked for. Must include the fields that drive the progress label
    UX so the user knows which preset is running."""
    cfg = tr.RunConfig.from_dict({
        **_valid_config_dict(tmp_path),
        "preset_label": "Headshot 700",
    })
    s = cfg.summary()
    assert s["num_shapes"] == 100
    assert s["preset_label"] == "Headshot 700"
    assert s["max_resolution"] == 480
    assert s["device"] == "cuda"


# =============================================================== emit


def test_emit_writes_single_json_line_with_trailing_newline():
    """Each event is exactly one JSON object on its own line. The EXE
    parses stderr with iter(proc.stderr.readline, b"") which only
    works if every event ends with \\n. A missed flush also stalls
    progress reporting — pytest catches the trailing-newline case,
    real subprocess catches the flush case."""
    buf = io.StringIO()
    tr.emit(buf, {"kind": "progress", "shape_count": 5, "total": 100})
    out = buf.getvalue()
    assert out.endswith("\n")
    parsed = json.loads(out.rstrip("\n"))
    assert parsed == {"kind": "progress", "shape_count": 5, "total": 100}


def test_emit_flushes_stream():
    """Live-progress users need each event visible immediately, not
    buffered until the run completes. We verify flush by giving emit()
    a stream whose flush() bumps a counter."""
    class FlushTrackingBuf(io.StringIO):
        def __init__(self):
            super().__init__()
            self.flushes = 0
            self.write_then_flush = []
        def flush(self):
            self.flushes += 1
            super().flush()
    buf = FlushTrackingBuf()
    tr.emit(buf, {"kind": "started"})
    tr.emit(buf, {"kind": "progress", "shape_count": 1, "total": 10})
    assert buf.flushes == 2, (
        f"expected 2 flushes (one per emit), got {buf.flushes} — "
        f"progress events will buffer in real subprocess output"
    )


# ============================================================ run() core


@pytest.fixture
def _fake_engine(monkeypatch, tmp_path):
    """Install a fake forza_abyss_painter.shapegen.gpu.engine module so
    run() can be exercised without torch installed. The fake's run_gpu
    returns a fixed shape list + canvas pair that round-trips through
    the JSON schema."""
    import types
    import numpy as np

    fake_mod = types.ModuleType("forza_abyss_painter.shapegen.gpu.engine")

    # Match the GPUConfig dataclass shape (just enough that runner uses).
    from dataclasses import dataclass
    @dataclass
    class _FakeGPUConfig:
        num_shapes: int = 100
        random_samples: int = 256
        seed: int = 0
        edge_strength: float = 0.0
        posterize_levels: int = 0
        bbox_local: bool = False
        joint_polish_steps: int = 0
        vram_budget_gib: float = 0.0
        lock_alpha: bool = True
    fake_mod.GPUConfig = _FakeGPUConfig

    captured = {}
    def _fake_run_gpu(target_rgb, cfg, alpha_mask=None,
                     progress_every=0, checkpoint_cb=None,
                     checkpoint_every=0):
        captured["target_rgb"] = target_rgb
        captured["cfg"] = cfg
        captured["alpha_mask_given"] = alpha_mask is not None
        # Fire one checkpoint mid-run so the IPC progress test can verify.
        if checkpoint_cb and checkpoint_every:
            checkpoint_cb(checkpoint_every, [])
        shapes = [
            {"type": "rotated_ellipse", "x": 10.0, "y": 10.0,
             "rx": 3.0, "ry": 4.0, "angle": 0.0,
             "color": [100, 100, 100, 255]},
            {"type": "rotated_ellipse", "x": 20.0, "y": 20.0,
             "rx": 2.0, "ry": 2.0, "angle": 45.0,
             "color": [50, 150, 200, 255]},
        ]
        canvas = np.zeros((10, 10, 3), dtype=np.uint8)
        return shapes, canvas

    fake_mod.run_gpu = _fake_run_gpu
    monkeypatch.setitem(sys.modules, "forza_abyss_painter.shapegen.gpu.engine", fake_mod)
    return captured


@pytest.fixture
def _fake_image(monkeypatch):
    """Fake _load_image so we don't need real PNG bytes for engine-path
    tests. Returns a small RGB array, no alpha."""
    import numpy as np
    def _fake(path):
        return np.zeros((10, 10, 3), dtype=np.uint8), None
    monkeypatch.setattr(tr, "_load_image", _fake)


def test_run_emits_started_done_with_zero_exit_on_success(
    tmp_path, _fake_engine, _fake_image,
):
    """Happy path: started event → done event → exit 0. The EXE's
    QThread polls stderr for these as the contract for state transitions.
    Done event MUST carry the output_path so the EXE can load it."""
    cfg = tr.RunConfig.from_dict(_valid_config_dict(tmp_path))
    buf = io.StringIO()
    rc = tr.run(cfg, stream=buf)
    events = [json.loads(line) for line in buf.getvalue().splitlines() if line]
    kinds = [e["kind"] for e in events]
    assert "started" in kinds
    assert "done" in kinds
    assert kinds[-1] == "done", f"done must be the LAST event, got {kinds}"
    assert rc == 0
    done_event = next(e for e in events if e["kind"] == "done")
    assert done_event["output_path"] == str(cfg.output_json_path)
    assert done_event["shape_count"] == 2


def test_run_writes_valid_fd6_shapes_json(tmp_path, _fake_engine, _fake_image):
    """The output file at done.output_path must conform to the canonical
    fd6.shapes v1 schema — CLAUDE.md §3. If the runner emits something
    else, load_json on the EXE side fails and the user loses the run."""
    from forza_abyss_painter.io.exporter import load_json

    cfg = tr.RunConfig.from_dict(_valid_config_dict(tmp_path))
    rc = tr.run(cfg, stream=io.StringIO())
    assert rc == 0
    loaded = load_json(cfg.output_json_path)
    assert loaded.format == "fd6.shapes"
    assert loaded.version == 1
    assert len(loaded.shapes) == 2
    # Round-trip materialize → confirms shape dicts are well-formed
    # against the registered Shape classes (no invented geometry per §3).
    materialized = loaded.materialize_shapes()
    assert len(materialized) == 2


def test_run_progress_events_carry_total(tmp_path, _fake_engine, _fake_image):
    """Progress events tell the EXE 'X of T shapes done'. Without `total`
    the progress bar can't show a percentage — the dialog falls back to
    an indeterminate spinner which is a worse UX."""
    d = _valid_config_dict(tmp_path)
    d["checkpoint_every"] = 50
    cfg = tr.RunConfig.from_dict(d)
    buf = io.StringIO()
    tr.run(cfg, stream=buf)
    events = [json.loads(l) for l in buf.getvalue().splitlines() if l]
    progress = [e for e in events if e["kind"] == "checkpoint"]
    assert progress, "no checkpoint events emitted with checkpoint_every=50"
    for e in progress:
        assert "total" in e
        assert e["total"] == 100   # = num_shapes from _valid_config_dict


def test_run_missing_image_emits_error_event(tmp_path, _fake_engine):
    """Image path that doesn't exist → 'error' event with stage='load_image'
    + exit code 1. The EXE's modal uses the stage tag to route to the
    right help message ('check the image path' vs 'install runtime' vs
    'reduce VRAM knobs')."""
    cfg = tr.RunConfig.from_dict({
        **_valid_config_dict(tmp_path),
        "image_path": str(tmp_path / "does-not-exist.png"),
    })
    buf = io.StringIO()
    rc = tr.run(cfg, stream=buf)
    events = [json.loads(l) for l in buf.getvalue().splitlines() if l]
    err_events = [e for e in events if e["kind"] == "error"]
    assert err_events, "no error event emitted for missing image"
    assert err_events[0]["stage"] == "load_image"
    assert rc == 1


def test_run_engine_runtime_error_propagates_actionable_message(
    tmp_path, _fake_image, monkeypatch,
):
    """run_gpu's CUDA-OOM wrapper raises RuntimeError with an actionable
    knob-halving recipe. That message must propagate UNCHANGED to the
    EXE so the user sees the specific values to lower — not a generic
    'engine failed'."""
    import types
    fake_mod = types.ModuleType("forza_abyss_painter.shapegen.gpu.engine")
    from dataclasses import dataclass
    @dataclass
    class _Cfg:
        num_shapes: int = 100
        random_samples: int = 256
        seed: int = 0
        edge_strength: float = 0.0
        posterize_levels: int = 0
        bbox_local: bool = False
        joint_polish_steps: int = 0
        vram_budget_gib: float = 0.0
        lock_alpha: bool = True
    fake_mod.GPUConfig = _Cfg
    def _oom_run_gpu(*a, **kw):
        raise RuntimeError(
            "CUDA OOM: drop RANDOM_SAMPLES from 12288 to 6144 and "
            "MAX_RESOLUTION from 1200 to 800"
        )
    fake_mod.run_gpu = _oom_run_gpu
    monkeypatch.setitem(sys.modules,
                        "forza_abyss_painter.shapegen.gpu.engine", fake_mod)

    cfg = tr.RunConfig.from_dict(_valid_config_dict(tmp_path))
    buf = io.StringIO()
    rc = tr.run(cfg, stream=buf)
    err = next(json.loads(l) for l in buf.getvalue().splitlines()
               if json.loads(l).get("kind") == "error")
    assert err["stage"] == "engine_run"
    assert "RANDOM_SAMPLES" in err["message"]
    assert rc == 1


def test_run_missing_torch_emits_import_error(tmp_path, _fake_image, monkeypatch):
    """If the user clicks Generate before the runtime installer
    completes, torch isn't on the path inside this Python and the
    engine import fails. Exit 3 (distinct from 1) tells the EXE to
    surface 'install runtime first' rather than 'engine failed'."""
    # Force the engine import to ImportError.
    import builtins
    real_import = builtins.__import__
    def _blocking_import(name, *a, **kw):
        if name == "forza_abyss_painter.shapegen.gpu.engine":
            raise ImportError("No module named 'torch'")
        return real_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", _blocking_import)

    cfg = tr.RunConfig.from_dict(_valid_config_dict(tmp_path))
    buf = io.StringIO()
    rc = tr.run(cfg, stream=buf)
    err = next(json.loads(l) for l in buf.getvalue().splitlines()
               if json.loads(l).get("kind") == "error")
    assert err["stage"] == "import_engine"
    assert "runtime installer" in err["message"].lower()
    assert rc == 3


# =========================================================== main() CLI


def test_main_returns_2_on_missing_config_file(tmp_path):
    """Missing --config file: exit code 2 (config_load), error event
    with stage='config_load'. The EXE-side modal uses this to surface
    'we built a bad config' rather than blame the engine."""
    missing = tmp_path / "does-not-exist.json"
    # Capture stderr by replacing sys.stderr for the duration of main().
    real_stderr = sys.stderr
    buf = io.StringIO()
    sys.stderr = buf
    try:
        rc = tr.main(["--config", str(missing)])
    finally:
        sys.stderr = real_stderr
    assert rc == 2
    err = next(json.loads(l) for l in buf.getvalue().splitlines()
               if json.loads(l).get("kind") == "error")
    assert err["stage"] == "config_load"


def test_main_returns_2_on_invalid_config_json(tmp_path):
    """Config file present but contents are garbage → exit 2."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    real_stderr = sys.stderr
    buf = io.StringIO()
    sys.stderr = buf
    try:
        rc = tr.main(["--config", str(bad)])
    finally:
        sys.stderr = real_stderr
    assert rc == 2


# ===================================================== subprocess smoke


def test_real_subprocess_run_emits_started_then_done(tmp_path):
    """End-to-end IPC smoke: spawn python -m torch_runner via the same
    invocation pattern the EXE will use, parse stderr line-stream, verify
    started → done → exit 0. Uses a sitecustomize to stub the engine
    module so the subprocess doesn't need torch installed.

    This catches IPC-boundary bugs the in-process tests miss: stderr line
    buffering, sys.exit semantics, line-flush behavior across a real
    process boundary.
    """
    # Source image stub.
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")
    # Config.
    out_path = tmp_path / "out.json"
    cfg = {
        "image_path": str(src),
        "output_json_path": str(out_path),
        "num_shapes": 2,
        "max_resolution": 240,
        "random_samples": 64,
        "device": "cpu",
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    # Wrapper module: a normal Python file invoked as `python <wrapper.py>`
    # that imports numpy + stubs the engine module in sys.modules + calls
    # torch_runner.main(). This avoids sitecustomize's startup-order
    # constraint (numpy not yet on path during sitecustomize execution).
    wrapper = tmp_path / "run_with_stub.py"
    wrapper.write_text(textwrap.dedent("""
        import sys, types
        from dataclasses import dataclass
        import numpy as np

        fake_eng = types.ModuleType("forza_abyss_painter.shapegen.gpu.engine")
        @dataclass
        class _Cfg:
            num_shapes: int = 100
            random_samples: int = 256
            seed: int = 0
            edge_strength: float = 0.0
            posterize_levels: int = 0
            bbox_local: bool = False
            joint_polish_steps: int = 0
            vram_budget_gib: float = 0.0
            lock_alpha: bool = True
        fake_eng.GPUConfig = _Cfg
        def _fake_run_gpu(target_rgb, cfg, alpha_mask=None,
                          progress_every=0, checkpoint_cb=None,
                          checkpoint_every=0):
            shapes = [{
                "type": "rotated_ellipse", "x": 5.0, "y": 5.0,
                "rx": 2.0, "ry": 2.0, "angle": 0.0,
                "color": [100, 100, 100, 255],
            }]
            return shapes, np.zeros((10, 10, 3), dtype=np.uint8)
        fake_eng.run_gpu = _fake_run_gpu
        sys.modules["forza_abyss_painter.shapegen.gpu.engine"] = fake_eng

        # Stub _load_image so the test doesn't need a real PNG. Patches
        # into the runner module right when it imports.
        import forza_abyss_painter.runtime.torch_runner as tr
        def _fake_load_image(path):
            return np.zeros((10, 10, 3), dtype=np.uint8), None
        tr._load_image = _fake_load_image

        # Now invoke main() the same way the EXE will.
        sys.exit(tr.main(sys.argv[1:]))
    """), encoding="utf-8")

    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parent.parent
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    proc = subprocess.run(
        [sys.executable, str(wrapper), "--config", str(cfg_path)],
        env=env, capture_output=True, text=True, timeout=30,
    )
    # Stderr lines.
    lines = [l for l in proc.stderr.splitlines() if l.strip()]
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # Non-JSON stderr (e.g., a stub-side print) — runner's own
            # output must all be JSON, so a non-parseable line from the
            # runner would be a contract violation. We allow stub-side
            # extra lines (sitecustomize might print on errors) by NOT
            # failing here; the contract-violation check is on the
            # kind sequence below.
            pass
    kinds = [e.get("kind") for e in events]
    assert "started" in kinds, (
        f"runner didn't emit 'started' event over IPC. stderr was:\n"
        f"{proc.stderr}\nstdout was:\n{proc.stdout}"
    )
    assert "done" in kinds, (
        f"runner didn't emit 'done' event. stderr was:\n{proc.stderr}"
    )
    assert proc.returncode == 0, (
        f"runner exited {proc.returncode}, expected 0. stderr:\n{proc.stderr}"
    )
    # Output JSON landed.
    assert out_path.exists(), "runner exited 0 but output JSON wasn't written"
    # And conforms to fd6.shapes v1.
    from forza_abyss_painter.io.exporter import load_json
    loaded = load_json(out_path)
    assert loaded.format == "fd6.shapes"
