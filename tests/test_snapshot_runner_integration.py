"""Subprocess: fresh GPU run with checkpoint_every=100, num_shapes=300
must produce 3 snapshot files at the right names + each must parse +
contain _run_config."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")


def _image(path: Path, h=32, w=32):
    from PIL import Image
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :w // 2] = (200, 80, 80)
    arr[:, w // 2:] = (80, 80, 200)
    Image.fromarray(arr, "RGB").save(path)


@pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("FAP_SNAPSHOT_TEST_FORCE_CPU") != "1",
    reason="snapshot test runs the real runner; skipping on CPU-only host. "
           "Set FAP_SNAPSHOT_TEST_FORCE_CPU=1 to force.",
)
def test_runner_writes_snapshots_at_each_checkpoint(tmp_path):
    image = tmp_path / "img.png"
    _image(image)
    out = tmp_path / "out" / "fixture.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "image_path": str(image),
        "output_json_path": str(out),
        "num_shapes": 300,
        "max_resolution": 360,
        "random_samples": 256,
        "checkpoint_every": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lock_alpha": True,
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "forza_abyss_painter.runtime.torch_runner",
         "--config", str(cfg_path)],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"runner exited {result.returncode}\nstderr:\n{result.stderr}"
    )
    assert out.is_file()
    for n in (100, 200, 300):
        snap = out.parent / f"fixture_{n}.json"
        assert snap.is_file(), f"missing snapshot {snap}"
        doc = json.loads(snap.read_text(encoding="utf-8"))
        assert doc["format"] == "fd6.shapes"
        assert doc["shape_count"] == n
        assert "_run_config" in doc, f"snapshot {snap.name} missing _run_config"
        rc = doc["_run_config"]
        assert rc["target_shape_count"] == 300
        assert rc["random_samples"] == 256
        assert rc["max_resolution"] == 360


def test_snapshot_event_in_stderr(tmp_path):
    """The snapshot event must appear on stderr alongside checkpoint
    events. Runs without GPU (fewer shapes, small image)."""
    image = tmp_path / "img.png"
    _image(image, h=16, w=16)
    out = tmp_path / "out" / "fixture.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "image_path": str(image),
        "output_json_path": str(out),
        "num_shapes": 6,
        "max_resolution": 64,
        "random_samples": 16,
        "checkpoint_every": 3,
        "device": "cpu",
        "lock_alpha": True,
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "forza_abyss_painter.runtime.torch_runner",
         "--config", str(cfg_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0
    snapshot_events = []
    for line in result.stderr.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("kind") == "snapshot":
            snapshot_events.append(ev)
    assert len(snapshot_events) >= 1
    first = snapshot_events[0]
    assert "shape_count" in first
    assert "total" in first
    assert "path" in first
    assert Path(first["path"]).is_file()
