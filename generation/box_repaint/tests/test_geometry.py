from generation.box_repaint.geometry import round16, compute_resize, scale_bbox


def test_round16_rounds_to_nearest_multiple_of_16():
    assert round16(100) == 96
    assert round16(105) == 112


def test_round16_never_goes_below_16():
    assert round16(1) == 16
    assert round16(8) == 16


def test_compute_resize_produces_dimensions_divisible_by_16():
    new_w, new_h, sx, sy = compute_resize(350, 244, target_w=1024)
    assert new_w % 16 == 0
    assert new_h % 16 == 0
    assert new_w == 1024


def test_compute_resize_scale_factors_are_consistent_with_new_dims():
    orig_w, orig_h = 350, 244
    new_w, new_h, sx, sy = compute_resize(orig_w, orig_h, target_w=1024)
    assert abs(sx - new_w / orig_w) < 1e-9
    assert abs(sy - new_h / orig_h) < 1e-9


def test_scale_bbox_scales_all_four_components():
    result = scale_bbox((10.0, 20.0, 30.0, 40.0), 2.0, 0.5)
    assert result == (20.0, 10.0, 60.0, 20.0)
