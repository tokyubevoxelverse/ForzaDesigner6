from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from forza_abyss_painter.io.json_schema import FD6Document


def save_json(doc: FD6Document, path: str | Path) -> Path:
    """Atomic write: serialize to a sibling tempfile then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = doc.to_dict()
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def load_json(path: str | Path) -> FD6Document:
    path = Path(path)
    # utf-8-sig transparently skips the UTF-8 BOM (EF BB BF) if present
    # and is a no-op when it's absent. Notepad's default save on Windows
    # adds a BOM; without this, hand-edited JSONs error out before
    # reaching the schema validator. (Cursor Run 6 R6.3eA finding.)
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return FD6Document.from_dict(data)
