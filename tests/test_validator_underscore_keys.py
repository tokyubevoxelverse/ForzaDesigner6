"""The fd6.shapes validator must tolerate snapshot metadata fields
prefixed with `_`. `_run_config` is the resume breadcrumb but the
convention is open-ended: any `_*` top-level key is metadata, not
contract, and should not trigger warnings or errors.
"""
from __future__ import annotations

from forza_abyss_painter.io.validator import validate_document, Severity


def _good_doc() -> dict:
    return {
        "format": "fd6.shapes",
        "version": 1,
        "source_image": "x.png",
        "image_size": [64, 64],
        "shape_count": 1,
        "generated_at": "",
        "profile": "",
        "sticker_mode": False,
        "shapes": [
            {"type": "rotated_ellipse", "x": 16.0, "y": 32.0,
             "rx": 8.0, "ry": 8.0, "angle": 0.0,
             "color": [128, 128, 128, 255]},
        ],
    }


def test_baseline_clean_doc_has_no_errors():
    issues = validate_document(_good_doc())
    errors = [i for i in issues if i.severity is Severity.ERROR]
    assert errors == []


def test_run_config_block_does_not_trigger_warnings():
    doc = _good_doc()
    doc["_run_config"] = {
        "target_shape_count": 100,
        "random_samples": 1024,
        "max_resolution": 360,
    }
    issues = validate_document(doc)
    # No issue should mention `_run_config` as unknown/unexpected.
    for issue in issues:
        msg = (issue.message or "").lower()
        assert "_run_config" not in msg, (
            f"validator complained about _run_config: {issue}"
        )


def test_arbitrary_underscore_metadata_tolerated():
    """Other underscore-prefixed keys (future metadata) shouldn't warn."""
    doc = _good_doc()
    doc["_diag"] = {"wall_time_s": 17.3}
    doc["_my_experiment"] = "alpha"
    issues = validate_document(doc)
    for issue in issues:
        msg = (issue.message or "").lower()
        assert "_diag" not in msg
        assert "_my_experiment" not in msg


def test_real_field_typos_still_warn_or_error():
    """Defensive: only `_*` is metadata. A typo'd real field should
    still get flagged. E.g. `formatt` (typo of `format`)."""
    doc = _good_doc()
    doc["formatt"] = "fd6.shapes"   # typo, no underscore
    issues = validate_document(doc)
    errors = [i for i in issues if i.severity is Severity.ERROR]
    # The shapes are still well-formed; ERROR list should remain empty.
    assert errors == []
    # Positive check: the typo'd key must have been flagged at INFO.
    assert any(i.code == "unknown_top_level_field" for i in issues), (
        "validator should emit an unknown-field issue for 'formatt' typo"
    )
