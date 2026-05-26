from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QThread, Signal

from fd6.shapegen.engine import Engine, EngineConfig
from fd6.shapegen.line_guide import resolve_line_guide
from fd6.shapegen.profile import Profile
from fd6.io.exporter import save_json
from fd6.io.json_schema import FD6Document


def _posterize_rgb(target: np.ndarray, levels: int) -> np.ndarray:
    if levels >= 256:
        return target
    bounded_levels = max(2, int(levels))
    scale = bounded_levels - 1
    work = target.astype(np.uint16)
    quantized = (work * scale + 127) // 255
    return ((quantized * 255 + (scale // 2)) // scale).astype(np.uint8)


def _preprocess_target(target: np.ndarray, posterize_levels: int) -> np.ndarray:
    processed = _posterize_rgb(target, posterize_levels)
    luma = (
        processed[:, :, 0].astype(np.int32) * 77
        + processed[:, :, 1].astype(np.int32) * 150
        + processed[:, :, 2].astype(np.int32) * 29
    ) >> 8
    padded = np.pad(luma, ((1, 1), (1, 1)), mode="edge")
    neighborhood = (
        padded[1:-1, 1:-1] * 4
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    ) >> 3
    detail = luma - neighborhood
    boost_limit = 24 if posterize_levels < 128 else 16
    boost_gain = 2 if posterize_levels < 64 else 1
    boost = np.clip(detail * boost_gain, -boost_limit, boost_limit).astype(np.int16)
    return np.clip(processed.astype(np.int16) + boost[:, :, None], 0, 255).astype(np.uint8)


def _prepare_target_image(img: Image.Image, profile: Profile, sticker_mode: bool) -> tuple[np.ndarray, np.ndarray | None]:
    alpha_mask: np.ndarray | None = None
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        rgba = img.convert("RGBA")
        if sticker_mode:
            arr_rgba = np.asarray(rgba, dtype=np.uint8)
            img = Image.fromarray(arr_rgba[:, :, :3], "RGB")
            alpha_mask = arr_rgba[:, :, 3].copy()
        else:
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[3])
            img = bg
    else:
        img = img.convert("RGB")
    if not sticker_mode and img.size[0] != img.size[1]:
        side = max(img.size)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        offset = ((side - img.size[0]) // 2, (side - img.size[1]) // 2)
        square.paste(img, offset)
        img = square
    mr = profile.max_resolution
    if max(img.size) > mr:
        scale = mr / max(img.size)
        new_size = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
        img = img.resize(new_size, Image.LANCZOS)
        if alpha_mask is not None:
            am_img = Image.fromarray(alpha_mask, "L").resize(new_size, Image.LANCZOS)
            alpha_mask = np.asarray(am_img, dtype=np.uint8)
    target = np.asarray(img, dtype=np.uint8)
    return _preprocess_target(target, profile.posterize_levels), alpha_mask


def _prepare_line_guide_source_image(img: Image.Image, profile: Profile, sticker_mode: bool) -> Image.Image:
    if sticker_mode:
        work = img.convert("RGB")
    else:
        rgba = img.convert("RGBA") if (
            img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
        ) else None
        if rgba is not None:
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[3])
            work = bg
        else:
            work = img.convert("RGB")
        if work.size[0] != work.size[1]:
            side = max(work.size)
            square = Image.new("RGB", (side, side), (255, 255, 255))
            offset = ((side - work.size[0]) // 2, (side - work.size[1]) // 2)
            square.paste(work, offset)
            work = square
    mr = profile.max_resolution
    if max(work.size) > mr:
        scale = mr / max(work.size)
        new_size = (max(1, int(work.size[0] * scale)), max(1, int(work.size[1] * scale)))
        work = work.resize(new_size, Image.LANCZOS)
    return work


class GenerationWorker(QObject):
    """Wraps Engine.run() in a QThread-friendly object. Emits Qt signals for the GUI."""

    progress = Signal(int, int, float)  # shape_count, total, rms
    preview = Signal(object)            # np.ndarray (H,W,3) uint8
    backend_ready = Signal(str)
    line_guide_ready = Signal(str)
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
            target, alpha_mask = _prepare_target_image(img, self.profile, self.sticker_mode)
            line_source = _prepare_line_guide_source_image(img, self.profile, self.sticker_mode)
            line_guide = resolve_line_guide(
                line_source,
                (target.shape[1], target.shape[0]),
                self.profile,
                self.image_path,
                guide_source_size=img.size,
                pad_guide_to_square=not self.sticker_mode,
            )
            self.line_guide_ready.emit(line_guide.message)

            self._engine = Engine(
                target,
                EngineConfig(profile=self.profile),
                alpha_mask=alpha_mask,
                line_guide=line_guide.guide,
                line_guide_status=line_guide.message,
            )
            self.backend_ready.emit(self._engine.compute_label)
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
