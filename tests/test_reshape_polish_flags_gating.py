"""Pin Tier-B feature flags at False during development.

Mirrors tests/test_gpu_phase3_flag.py — the flag value cannot drift
independently of the plumbing landing. When the smoke is green and the
buttons are wired end-to-end, flip the flag in the SAME commit and
update this test (or delete it once stable, like GPU_PHASE_3).

Plumbing checklist for RESHAPE_GEN_AVAILABLE → True:
  - upload_panel emits `reshape_requested` on click
  - main_window slot `_on_reshape_requested` constructs GenerateLocallyDialog
    with `initial_source_path` from the loaded JSON
  - same-folder heuristic + picker fallback for source resolution
  - local smoke: real MainWindow loads a real JSON, clicks the button,
    fresh-gen completes end-to-end, output JSON lands + validates clean

Plumbing checklist for POLISH_LOADED_AVAILABLE → True:
  - upload_panel emits `polish_requested` on click
  - PolishDialog exposes (steps, lock_alpha, output_path) via .values()
  - torch_runner.RunConfig.mode == "polish_only" branch implemented
  - gpu_gen_worker.build_polish_config() writes valid config
  - local smoke: real MainWindow loads a real JSON, clicks the button,
    polish completes end-to-end, output _polished.json lands + validates
"""
from forza_abyss_painter.gui import feature_flags


def test_reshape_gen_flag_default_is_false():
    assert feature_flags.RESHAPE_GEN_AVAILABLE is False, (
        "Don't flip RESHAPE_GEN_AVAILABLE until the plumbing checklist "
        "in this test's docstring is fully landed and smoke-tested."
    )


def test_polish_loaded_flag_default_is_false():
    assert feature_flags.POLISH_LOADED_AVAILABLE is False, (
        "Don't flip POLISH_LOADED_AVAILABLE until the plumbing checklist "
        "in this test's docstring is fully landed and smoke-tested."
    )
