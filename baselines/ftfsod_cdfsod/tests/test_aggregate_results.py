import json
import statistics

from baselines.ftfsod_cdfsod.aggregate_results import load_results, summarize


def test_load_and_summarize_groups_by_domain_and_shot(tmp_path):
    results_dir = tmp_path
    (results_dir / "swinB_FISH_1shot_seed42.json").write_text(
        json.dumps({"coco/bbox_mAP": 0.30}))
    (results_dir / "swinB_FISH_1shot_seed43.json").write_text(
        json.dumps({"coco/bbox_mAP": 0.34}))
    (results_dir / "swinB_ArTaxOr_1shot_seed42.json").write_text(
        json.dumps({"coco/bbox_mAP": 0.50}))

    results = load_results(str(results_dir))

    assert results[("FISH", "1")] == [30.0, 34.0]
    assert results[("ArTaxOr", "1")] == [50.0]

    rows = summarize(results)
    fish_row = next(r for r in rows if r["domain"] == "FISH" and r["shot"] == "1")
    assert fish_row["n"] == 2
    assert fish_row["mean"] == statistics.mean([30.0, 34.0])
    assert fish_row["std"] == statistics.stdev([30.0, 34.0])

    artaxor_row = next(r for r in rows if r["domain"] == "ArTaxOr" and r["shot"] == "1")
    assert artaxor_row["n"] == 1
    assert artaxor_row["std"] == 0.0
