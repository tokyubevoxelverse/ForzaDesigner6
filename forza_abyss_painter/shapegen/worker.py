from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QThread, Signal

from forza_abyss_painter.shapegen.engine import Engine, EngineConfig
from forza_abyss_painter.shapegen.profile import Profile
from forza_abyss_painter.io.exporter import save_json
from forza_abyss_painter.io.json_schema import FD6Document


class GenerationWorker(QObject):
    """Wraps Engine.run() in a QThread-friendly object. Emits Qt signals for the GUI."""

    progress = Signal(int, int, float)  # shape_count, total, rms
    preview = Signal(object)            # np.ndarray (H,W,3) uint8
    finished = Signal(str)              # final json output path
    error = Signal(str)
    checkpoint_written = Signal(str)    # checkpoint json path

    def __init__(self, image_path: Path, profile: Profile, output_dir: Path | None = None, sticker_mode: bool = False) -> None:
        super().__init__()
        self.image_path = Path(image_path)
        self.profile = profile
        self.output_dir = Path(output_dir) if output_dir else self.image_path.parent / self.image_path.stem
        self.sticker_mode = sticker_mode  # When True, keep source alpha and skip transparent areas
        self._engine: Engine | None = None
        self._paused = False

    def stop(self) -> None:
        if self._engine:
            self._engine.request_stop()

    def set_pause(self, paused: bool) -> None:
        self._paused = paused
        if self._engine:
            self._engine.set_pause(paused)

    def run(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            img = Image.open(self.image_path)
            alpha_mask: np.ndarray | None = None  # None = full opacity (treat all pixels equally)
            has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
            if has_alpha:
                rgba = img.convert("RGBA")
                if self.sticker_mode:
                    # Keep transparency: extract alpha mask, use RGB channels as target
                    # (transparent areas keep whatever RGB they had — we ignore them via the mask)
                    arr_rgba = np.asarray(rgba, dtype=np.uint8)
                    img = Image.fromarray(arr_rgba[:, :, :3], "RGB")
                    alpha_mask = arr_rgba[:, :, 3].copy()  # H x W, 0 = transparent, 255 = opaque
                else:
                    # Default: composite onto white to avoid leaking under-transparent RGB junk
                    bg = Image.new("RGB", rgba.size, (255, 255, 255))
                    bg.paste(rgba, mask=rgba.split()[3])
                    img = bg
            else:
                img = img.convert("RGB")
            # When "Add white background" mode is active (sticker_mode False), also
            # pad non-square images to a square white canvas. This makes the FH6
            # vinyl-group canvas (which is square) fill cleanly with white outside
            # the original image rect, instead of leaving transparent strips.
            if not self.sticker_mode and img.size[0] != img.size[1]:
                side = max(img.size)
                square = Image.new("RGB", (side, side), (255, 255, 255))
                offset = ((side - img.size[0]) // 2, (side - img.size[1]) // 2)
                square.paste(img, offset)
                img = square

            # Edge-buffer padding (applied to every generation, transparent or not).
            # If any of the source pixels run all the way to the canvas edge, FH6's
            # vinyl renderer treats shapes whose extents touch that edge as
            # unbounded, which produces large smears and corner artifacts after
            # injection. Padding the source by ~8% on each side gives the shape
            # generator room so even shapes that land on the outermost rows/cols
            # of the *content* stay several pixels away from the actual canvas
            # edge once we hand off to the engine.
            BUFFER_FRAC = 0.08  # 8% per side → 16% larger output canvas
            pad_px = max(8, int(round(max(img.size) * BUFFER_FRAC)))
            new_w = img.size[0] + 2 * pad_px
            new_h = img.size[1] + 2 * pad_px
            if self.sticker_mode:
                buffered = Image.new("RGB", (new_w, new_h), (0, 0, 0))
                buffered.paste(img, (pad_px, pad_px))
                img = buffered
                if alpha_mask is not None:
                    padded_alpha = np.zeros((new_h, new_w), dtype=np.uint8)
                    src_h, src_w = alpha_mask.shape[:2]
                    padded_alpha[pad_px:pad_px + src_h, pad_px:pad_px + src_w] = alpha_mask
                    alpha_mask = padded_alpha
            else:
                buffered = Image.new("RGB", (new_w, new_h), (255, 255, 255))
                buffered.paste(img, (pad_px, pad_px))
                img = buffered
            # Downscale to profile.max_resolution along the longer side.
            mr = self.profile.max_resolution
            if max(img.size) > mr:
                scale = mr / max(img.size)
                new_size = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
                img = img.resize(new_size, Image.LANCZOS)
                if alpha_mask is not None:
                    am_img = Image.fromarray(alpha_mask, "L").resize(new_size, Image.LANCZOS)
                    alpha_mask = np.asarray(am_img, dtype=np.uint8)
            target = np.asarray(img, dtype=np.uint8)

            self._engine = Engine(target, EngineConfig(profile=self.profile), alpha_mask=alpha_mask)
            stem = self.image_path.stem
            final_path = self.output_dir / f"{stem}.json"

            for event in self._engine.run():
                if event.kind == "shape_committed":
                    self.progress.emit(event.shape_count, self.profile.stop_at, event.rms)
                elif event.kind == "preview" and event.canvas is not None:
                    self.preview.emit(event.canvas)
                elif event.kind == "checkpoint":
                    cp_path = self.output_dir / f"{stem}_{event.shape_count}.json"
                    doc = FD6Document.from_engine(
                        source_image=self.image_path.name,
                        image_size=(target.shape[1], target.shape[0]),
                        shapes=self._engine.shapes,
                        profile_name=self.profile.name,
                        sticker_mode=self.sticker_mode,
                    )
                    save_json(doc, cp_path)
                    self.checkpoint_written.emit(str(cp_path))
                elif event.kind == "error":
                    self.error.emit(event.message)
                    return
                elif event.kind == "done":
                    doc = FD6Document.from_engine(
                        source_image=self.image_path.name,
                        image_size=(target.shape[1], target.shape[0]),
                        shapes=self._engine.shapes,
                        profile_name=self.profile.name,
                        sticker_mode=self.sticker_mode,
                    )
                    save_json(doc, final_path)
                    self.finished.emit(str(final_path))
                    return
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
