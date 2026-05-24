import math

import pytest
from PIL import Image

from forza_abyss_painter.shapegen.cards import CARDS, usable_gib
from forza_abyss_painter.cli.fitsize import output_name, peak_bytes, plan_size, preset_params, resize_to
from forza_abyss_painter.shapegen.presets import PRESETS


def test_card_registry_has_expected_cards():
    assert set(CARDS) == {"blackwell-96", "a100-80", "a100-40", "l4-24", "t4-16", "v100-16"}
    assert CARDS["blackwell-96"] == 94
    assert CARDS["a100-80"] == 80


def test_usable_gib_known_card():
    assert usable_gib("a100-40") == 40


def test_usable_gib_unknown_card_lists_valid_names():
    with pytest.raises(ValueError) as exc:
        usable_gib("h100-80")
    msg = str(exc.value)
    assert "h100-80" in msg
    assert "a100-80" in msg  # error lists valid names


def test_peak_bytes_calibration():
    # 5 * K * L^2 ; the constant matches the observed ~80 GB at 1600px / 6144 samples.
    assert peak_bytes(1600, 6144) == 78_643_200_000.0


def test_plan_size_quality_bound():
    # Big source, big card -> the sqrt(N)*28 detail ceiling binds.
    target, binding, caps = plan_size(
        source_long=4000, num_shapes=3000, random_samples=6144, usable_gib=94
    )
    assert binding == "quality"
    assert target == int(math.sqrt(3000) * 28)  # 1533


def test_plan_size_device_bound():
    # Small card -> VRAM fit binds below the quality ceiling.
    target, binding, caps = plan_size(
        source_long=4000, num_shapes=3000, random_samples=6144, usable_gib=20
    )
    assert binding == "device"
    assert target < int(math.sqrt(3000) * 28)


def test_plan_size_source_bound():
    # Small source -> the source caps it (never upscale).
    target, binding, caps = plan_size(
        source_long=600, num_shapes=3000, random_samples=6144, usable_gib=94
    )
    assert binding == "source"
    assert target == 600


def test_preset_params_lockstep_with_presets():
    # fitsize must read N/K straight from the frozen baseline, no duplication.
    assert preset_params("highres_3000") == (
        PRESETS["highres_3000"]["num_shapes"],
        PRESETS["highres_3000"]["random_samples"],
    )


def test_output_name_is_self_documenting():
    name = output_name("holier_12h48m37s447", "highres_3000", "blackwell-96", 1533)
    assert name == "holier_12h48m37s447_highres3000_blackwell96_1533px.png"


def test_resize_to_downscales_preserving_aspect():
    img = Image.new("RGBA", (1600, 1000))
    out, changed = resize_to(img, 800)
    assert changed is True
    assert max(out.size) == 800
    assert out.size == (800, 500)  # aspect preserved


def test_resize_to_never_upscales():
    img = Image.new("RGBA", (400, 300))
    out, changed = resize_to(img, 1533)
    assert changed is False
    assert out.size == (400, 300)


def test_main_writes_sized_file_into_fit_subdir(tmp_path):
    from forza_abyss_painter.cli.fitsize import main

    src = tmp_path / "pose.png"
    Image.new("RGBA", (2000, 1200), (10, 20, 30, 255)).save(src)

    rc = main([str(src), "--preset", "highres_3000", "--card", "blackwell-96"])
    assert rc == 0

    out_dir = tmp_path / "fit"
    written = list(out_dir.glob("*.png"))
    assert len(written) == 1
    # highres_3000 on blackwell-96 is quality-bound at 1533px.
    assert written[0].name == "pose_highres3000_blackwell96_1533px.png"
    assert max(Image.open(written[0]).size) == 1533


def test_main_batches_a_directory(tmp_path):
    from forza_abyss_painter.cli.fitsize import main

    for i in range(3):
        Image.new("RGBA", (2000, 1200)).save(tmp_path / f"p{i}.png")

    rc = main([str(tmp_path), "--preset", "medium_1000", "--card", "t4-16"])
    assert rc == 0
    assert len(list((tmp_path / "fit").glob("*.png"))) == 3


def test_main_unknown_card_errors(tmp_path):
    from forza_abyss_painter.cli.fitsize import main

    src = tmp_path / "pose.png"
    Image.new("RGBA", (800, 600)).save(src)
    rc = main([str(src), "--preset", "highres_3000", "--card", "h100-80"])
    assert rc != 0


def test_main_nonexistent_path_errors(tmp_path):
    from forza_abyss_painter.cli.fitsize import main

    missing = tmp_path / "does_not_exist.png"  # never created
    rc = main([str(missing), "--preset", "highres_3000", "--card", "blackwell-96"])
    assert rc != 0
