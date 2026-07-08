import json

from experiments.clipart1k_box_repaint.compare_results import load_cell, summarize, compare, render_report


def test_load_cell_filters_by_domain_shot_and_tag(tmp_path):
    (tmp_path / "swinB_clipart1k_1shot_seed42.json").write_text(
        json.dumps({"coco/bbox_mAP": 0.50}))
    (tmp_path / "swinB_clipart1k_1shot_seed42_boxrepaint.json").write_text(
        json.dumps({"coco/bbox_mAP": 0.55}))
    (tmp_path / "swinB_FISH_1shot_seed42.json").write_text(
        json.dumps({"coco/bbox_mAP": 0.30}))

    baseline = load_cell(str(tmp_path), "clipart1k", "1", tag="")
    augmented = load_cell(str(tmp_path), "clipart1k", "1", tag="_boxrepaint")

    assert baseline == [50.0]
    assert augmented == [55.0]


def test_summarize_computes_mean_and_std():
    result = summarize([50.0, 52.0, 54.0])
    assert result["n"] == 3
    assert result["mean"] == 52.0
    assert round(result["std"], 4) == 2.0


def test_summarize_single_value_has_zero_std():
    result = summarize([50.0])
    assert result["std"] == 0.0


def test_compare_flags_signal_when_delta_exceeds_threshold():
    result = compare(baseline_values=[50.0, 50.5, 49.5],
                      augmented_values=[55.0, 55.5, 54.5])
    assert result["signal"] is True
    assert result["delta"] > 0


def test_compare_no_signal_when_delta_within_noise():
    result = compare(baseline_values=[50.0, 45.0, 55.0],
                      augmented_values=[50.5, 45.5, 55.5])
    assert result["signal"] is False


def test_render_report_states_the_gate_outcome():
    primary = compare([50.0, 50.5, 49.5], [55.0, 55.5, 54.5])
    secondary = compare([60.0, 60.5, 59.5], [55.0, 55.5, 54.5])
    report = render_report(primary, secondary)
    assert "PASS" in report
    assert "Stage 2" in report
