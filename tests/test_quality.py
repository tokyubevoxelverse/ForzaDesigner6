import random

import numpy as np

from fd6.shapegen.engine import _edge_candidate_shape
from fd6.shapegen.quality import (
    build_quality_context,
    edge_f1,
    edge_precision_recall_f1,
    gradient_error,
    precompute_weighted_rgb_error,
    precompute_gradient_error,
    ssim_index,
)
from fd6.shapegen.scoring import precompute_canvas_error, score_shape
from fd6.shapegen.shapes.rectangle import Rectangle


def _step_target(size=16):
    target = np.zeros((size, size, 3), dtype=np.uint8)
    target[:, size // 2:] = 255
    return target


def test_quality_context_weights_edges_and_respects_alpha():
    target = _step_target(12)
    alpha = np.full((12, 12), 255, dtype=np.uint8)
    alpha[9:, :] = 0

    context = build_quality_context(
        target,
        alpha,
        edge_weight_strength=2.0,
        gradient_weight=0.2,
        edge_alpha=224,
    )

    assert context is not None
    assert context.has_edge_points
    assert context.edge_weight[5, 6] > context.edge_weight[5, 1]
    assert context.edge_weight[10, 6] == 0.0
    assert context.edge_alpha == 224


def test_gradient_error_is_lower_for_matching_edge_than_flat_image():
    target = _step_target(16)
    flat = np.full_like(target, 127)
    matching = target.copy()
    context = build_quality_context(
        target,
        None,
        edge_weight_strength=1.0,
        gradient_weight=0.2,
        edge_alpha=224,
    )

    assert context is not None
    flat_total, flat_norm = precompute_gradient_error(flat, context)
    matching_total, matching_norm = precompute_gradient_error(matching, context)

    assert flat_norm == matching_norm
    assert gradient_error(matching, context) < gradient_error(flat, context)
    assert matching_total < flat_total


def test_edge_f1_and_ssim_reward_matching_structure():
    target = _step_target(16)
    shifted = np.zeros_like(target)
    shifted[:, 10:] = 255

    assert edge_f1(target, target) == 1.0
    assert edge_f1(shifted, target) < 1.0
    assert ssim_index(target, target) > ssim_index(shifted, target)


def test_edge_precision_recall_f1_responds_to_tolerance_and_missing_edges():
    target = _step_target(16)
    shifted = np.zeros_like(target)
    shifted[:, 9:] = 255
    flat = np.full_like(target, 127)

    p0, r0, f0 = edge_precision_recall_f1(shifted, target, tolerance=0)
    p1, r1, f1 = edge_precision_recall_f1(shifted, target, tolerance=1)
    p_missing, r_missing, f_missing = edge_precision_recall_f1(flat, target, tolerance=1)

    assert f1 >= f0
    assert p1 >= p0
    assert r1 >= r0
    assert (p_missing, r_missing, f_missing) == (0.0, 0.0, 0.0)


def test_edge_candidate_shape_uses_allowed_type_and_edge_alpha():
    target = _step_target(24)
    context = build_quality_context(
        target,
        None,
        edge_weight_strength=1.0,
        gradient_weight=0.1,
        edge_alpha=240,
    )

    assert context is not None
    shape = _edge_candidate_shape(
        random.Random(10),
        24,
        24,
        ["rotated_ellipse", "rectangle", "triangle"],
        context,
    )

    assert shape is not None
    assert shape.type_name in {"rotated_ellipse", "rectangle", "triangle"}
    assert shape.color[3] == 240


def test_line_guide_agreement_downweights_unsupported_lines():
    target = np.full((16, 16, 3), 128, dtype=np.uint8)
    guide = np.zeros((16, 16), dtype=np.float32)
    guide[:, 8] = 1.0

    loose = build_quality_context(
        target,
        None,
        edge_weight_strength=0.0,
        gradient_weight=0.0,
        edge_alpha=224,
        line_guide=guide,
        line_guide_strength=2.0,
        line_guide_agreement=0.0,
    )
    strict = build_quality_context(
        target,
        None,
        edge_weight_strength=0.0,
        gradient_weight=0.0,
        edge_alpha=224,
        line_guide=guide,
        line_guide_strength=2.0,
        line_guide_agreement=1.0,
    )

    assert loose is not None
    assert strict is not None
    assert loose.edge_weight[8, 8] > strict.edge_weight[8, 8]
    assert not strict.has_edge_points


def test_score_shape_stores_rms_and_quality_state_separately():
    target = _step_target(16)
    current = np.full_like(target, 127)
    context = build_quality_context(
        target,
        None,
        edge_weight_strength=1.0,
        gradient_weight=0.2,
        edge_alpha=255,
    )
    assert context is not None
    shape = Rectangle(color=(0, 0, 0, 255), x=3.5, y=8.0, hw=4.0, hh=8.0)
    full_sq, norm = precompute_canvas_error(current, target)
    weighted_sq, weighted_norm = precompute_weighted_rgb_error(current, target, context)
    gradient_sq, gradient_norm = precompute_gradient_error(current, context)

    plain_score, _ = score_shape(
        shape,
        current,
        target,
        canvas_full_sq=full_sq,
        canvas_norm=norm,
    )
    quality_score, _ = score_shape(
        shape,
        current,
        target,
        canvas_full_sq=full_sq,
        canvas_norm=norm,
        quality_context=context,
        weighted_full_sq=weighted_sq,
        weighted_norm=weighted_norm,
        gradient_full_error=gradient_sq,
        gradient_norm=gradient_norm,
    )

    assert np.isfinite(quality_score)
    assert np.isfinite(getattr(shape, "_fd6_rms_score"))
    assert getattr(shape, "_fd6_rms_score") == plain_score
    assert quality_score != plain_score
