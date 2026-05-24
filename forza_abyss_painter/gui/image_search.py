"""Image search + download panel — Google Images via embedded webview.

Hosts a Chromium-backed QWebEngineView pointed at Google Images. Downloads
triggered from the page's native right-click "Save image as..." gesture are
intercepted by `profile.downloadRequested` and routed through our PNG
conversion (Pillow) so the user gets a queue-ready PNG every time —
transparent PNGs keep their alpha.

Public API matches the old DDG-backed panel:
  - placed in UploadPanel's QStackedWidget as the "image searcher" tab
  - emits `image_downloaded(Path)` when a download completes
  - `shutdown()` is called by MainWindow.closeEvent

Theme integration: the toolbar/status line opt into the shared "ThemeGlow"
QSS objectName, matching Recents.
"""

from __future__ import annotations

import io
import re
import tempfile
import uuid
from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Qt, Signal
from PySide6.QtWebEngineCore import (
    QWebEngineDownloadRequest, QWebEnginePage, QWebEngineProfile,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)


GOOGLE_IMAGES_URL = "https://www.google.com/imghp?hl=en&tab=ri"


def _save_as_png_from_path(src: Path, dest_dir: Path, stem_hint: str) -> Path:
    """Re-encode a downloaded image file as a PNG that preserves transparency."""
    from PIL import Image
    img = Image.open(src)
    has_alpha = (
        img.mode in ("RGBA", "LA")
        or (img.mode == "P" and "transparency" in img.info)
    )
    if has_alpha and img.mode != "RGBA":
        img = img.convert("RGBA")
    elif img.mode == "CMYK":
        img = img.convert("RGB")
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", stem_hint)[:48] or "image"
    out = dest_dir / f"{safe}_{uuid.uuid4().hex[:8]}.png"
    img.save(out, "PNG")
    return out


class ImageSearchPanel(QWidget):
    """Google-Images-in-a-window with auto-PNG download."""

    image_downloaded = Signal(object)  # Path

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Inherit theme-glow background like Recents
        self.setObjectName("ThemeGlow")
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        # Quick search bar — wraps a Google Images search URL.
        bar = QHBoxLayout()
        self.query = QLineEdit(self)
        self.query.setPlaceholderText("Search Google Images…")
        self.query.returnPressed.connect(self._on_search_clicked)
        bar.addWidget(self.query, stretch=1)
        self.search_btn = QPushButton("Search", self)
        self.search_btn.clicked.connect(self._on_search_clicked)
        bar.addWidget(self.search_btn)
        self.home_btn = QPushButton("Home", self)
        self.home_btn.setToolTip("Back to images.google.com")
        self.home_btn.clicked.connect(self._on_home_clicked)
        bar.addWidget(self.home_btn)
        v.addLayout(bar)

        # Always-visible instruction banner. Stays put so users never miss
        # the right-click-to-save workflow even after multiple downloads.
        self.hint = QLabel(
            "TIP: right-click any image → \"Save image as…\" — it'll be "
            "auto-converted to PNG and added to your queue (transparency "
            "preserved).",
            self,
        )
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(
            "QLabel { background: rgba(255, 200, 60, 0.22);"
            " border: 1px solid rgba(255, 200, 60, 0.7);"
            " color: #ffe28a;"
            " padding: 6px 8px; border-radius: 6px;"
            " font-size: 11px; font-weight: bold; }"
        )
        v.addWidget(self.hint)

        # Separate transient status line for download progress / errors. The
        # banner above never changes so the instruction stays readable.
        self.status = QLabel("", self)
        self.status.setWordWrap(True)
        self.status.setStyleSheet("font-size: 11px; color: #c8c8c8;")
        v.addWidget(self.status)

        # Dedicated off-the-record profile so we don't share cookies with the
        # rest of the system / persist between launches.
        #
        # Mobile User-Agent forces Google to serve its narrow-viewport layout
        # — without this the desktop site renders in our ~300-px panel and
        # everything either clips or feels zoomed-out.
        self._profile = QWebEngineProfile(self)
        self._profile.setHttpUserAgent(
            "Mozilla/5.0 (Linux; Android 13; Pixel 7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Mobile Safari/537.36"
        )
        # Route every download through our PNG converter
        self._profile.downloadRequested.connect(self._on_download_requested)

        # Webview itself
        self.page = QWebEnginePage(self._profile, self)
        self.view = QWebEngineView(self)
        self.view.setPage(self.page)
        # Small zoom-out so the mobile-width layout fits the narrow sidebar
        # without horizontal scrollbars on most resolutions.
        self.view.setZoomFactor(0.9)
        self.view.setUrl(QUrl(GOOGLE_IMAGES_URL))
        v.addWidget(self.view, stretch=1)

        # Bookkeeping
        self._dest_dir = Path(tempfile.gettempdir()) / "fd6_image_downloads"
        self._dest_dir.mkdir(parents=True, exist_ok=True)
        # Track active downloads so we can post-process when they finish
        self._active: list[QWebEngineDownloadRequest] = []

    # ── navigation ───────────────────────────────────────────────────────

    def _on_search_clicked(self) -> None:
        q = self.query.text().strip()
        if not q:
            self.view.setUrl(QUrl(GOOGLE_IMAGES_URL))
            return
        # Use the URL-encoded query parameter so we never break special chars
        from urllib.parse import quote
        url = f"https://www.google.com/search?tbm=isch&q={quote(q)}&hl=en"
        self.view.setUrl(QUrl(url))

    def _on_home_clicked(self) -> None:
        self.view.setUrl(QUrl(GOOGLE_IMAGES_URL))

    # ── downloads ────────────────────────────────────────────────────────

    def _on_download_requested(self, dl: QWebEngineDownloadRequest) -> None:
        """Native page download → save raw to temp → re-encode as PNG."""
        # Pick a unique temp filename for the raw download
        suffix = Path(dl.suggestedFileName() or dl.url().fileName() or "img").suffix
        if not suffix or len(suffix) > 6:
            suffix = ".bin"
        raw_name = f"raw_{uuid.uuid4().hex[:10]}{suffix}"
        dl.setDownloadDirectory(str(self._dest_dir))
        dl.setDownloadFileName(raw_name)
        self._active.append(dl)
        dl.isFinishedChanged.connect(lambda d=dl: self._on_download_finished(d))
        dl.accept()
        self.status.setText("Downloading…")

    def _on_download_finished(self, dl: QWebEngineDownloadRequest) -> None:
        if not dl.isFinished():
            return
        # Use Path attributes instead of legacy path() to avoid deprecation
        raw_path = Path(dl.downloadDirectory()) / dl.downloadFileName()
        if not raw_path.exists():
            self.status.setText("Download finished but the file is missing.")
            return
        try:
            stem_hint = Path(dl.suggestedFileName() or "image").stem or "image"
            out = _save_as_png_from_path(raw_path, self._dest_dir, stem_hint)
        except Exception as exc:
            self.status.setText(f"Couldn't decode image: {type(exc).__name__}: {exc}")
            return
        finally:
            try:
                raw_path.unlink(missing_ok=True)
            except Exception:
                pass
            if dl in self._active:
                self._active.remove(dl)
        self.status.setText(f"Saved: {out.name} — added to queue.")
        self.image_downloaded.emit(out)

    # ── lifecycle ────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Stop the webview cleanly before MainWindow closes."""
        try:
            self.view.stop()
            self.view.setUrl(QUrl("about:blank"))
        except Exception:
            pass
