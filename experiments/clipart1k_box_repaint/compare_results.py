"""Compute the Stage-1 primary and secondary readings for the clipart1k
box-repaint rung-1 experiment (docs/specs/2026-07-08-clipart1k-box-repaint-rung1-design.md).

Primary: (1real+4synth) vs the frozen 1-shot baseline -- does augmentation
help at a fixed real-data budget? Decision gate: delta > 0 and
abs(delta) > 2 * baseline seed-std means "proceed to Stage 2".

Secondary: the same augmented cells vs the frozen 5-shot baseline -- can 4
synthetic variants substitute for 4 more real images? Expected to
underperform; reported to quantify the gap, not to gate anything.

Usage:
    python3 -m experiments.clipart1k_box_repaint.compare_results \
        --baseline-dir baselines/ftfsod_cdfsod/results \
        --augmented-dir experiments/clipart1k_box_repaint/results
"""
import argparse
import glob
import json
import os
import re
import statistics

RUN_NAME_RE = re.compile(
    r"^swinB_(?P<domain>.+)_(?P<shot>\d+)shot_seed(?P<seed>\d+)(?P<tag>_\w+)?$")


def load_cell(results_dir, domain, shot, tag=""):
    """Return the list of mAP percentages (one per seed) for one
    (domain, shot, tag) cell."""
    values = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        run_name = os.path.splitext(os.path.basename(path))[0]
        match = RUN_NAME_RE.match(run_name)
        if not match:
            continue
        if match.group("domain") != domain or match.group("shot") != shot:
            continue
        if (match.group("tag") or "") != tag:
            continue
        with open(path) as f:
            data = json.load(f)
        map_value = data.get("coco/bbox_mAP")
        if map_value is not None:
            values.append(round(map_value * 100, 4))
    return values


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def compare(baseline_values, augmented_values):
    baseline = summarize(baseline_values)
    augmented = summarize(augmented_values)
    delta = augmented["mean"] - baseline["mean"]
    threshold = 2 * baseline["std"]
    signal = abs(delta) > threshold
    return {
        "baseline": baseline,
        "augmented": augmented,
        "delta": delta,
        "threshold": threshold,
        "signal": signal,
    }


def render_report(primary, secondary):
    lines = []
    lines.append("## Stage 1 primary reading: augmentation vs 1-shot baseline")
    lines.append("")
    lines.append("| | mean +/- std | n |")
    lines.append("|---|---|---|")
    lines.append("| 1-shot baseline | {:.2f} +/- {:.2f} | {} |".format(
        primary["baseline"]["mean"], primary["baseline"]["std"], primary["baseline"]["n"]))
    lines.append("| 1real+4synth | {:.2f} +/- {:.2f} | {} |".format(
        primary["augmented"]["mean"], primary["augmented"]["std"], primary["augmented"]["n"]))
    lines.append("")
    lines.append("Delta = {:+.2f}, signal threshold (2x baseline std) = {:.2f}".format(
        primary["delta"], primary["threshold"]))
    gate = ("PASS -> proceed to Stage 2" if (primary["signal"] and primary["delta"] > 0)
            else "NO SIGNAL -> stop, do not run Stage 2")
    lines.append("Decision gate: {}".format(gate))
    lines.append("")
    lines.append("## Stage 1 secondary reading: augmentation vs 5-shot baseline (expected to underperform)")
    lines.append("")
    lines.append("| | mean +/- std | n |")
    lines.append("|---|---|---|")
    lines.append("| 5-shot baseline | {:.2f} +/- {:.2f} | {} |".format(
        secondary["baseline"]["mean"], secondary["baseline"]["std"], secondary["baseline"]["n"]))
    lines.append("| 1real+4synth | {:.2f} +/- {:.2f} | {} |".format(
        secondary["augmented"]["mean"], secondary["augmented"]["std"], secondary["augmented"]["n"]))
    lines.append("")
    lines.append("Delta = {:+.2f}".format(secondary["delta"]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default="baselines/ftfsod_cdfsod/results")
    parser.add_argument("--augmented-dir", default="experiments/clipart1k_box_repaint/results")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    augmented_1shot = load_cell(args.augmented_dir, "clipart1k", "1", tag="_boxrepaint")
    baseline_1shot = load_cell(args.baseline_dir, "clipart1k", "1")
    baseline_5shot = load_cell(args.baseline_dir, "clipart1k", "5")

    primary = compare(baseline_1shot, augmented_1shot)
    secondary = compare(baseline_5shot, augmented_1shot)

    report = render_report(primary, secondary)
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
