#!/usr/bin/env python3
"""Aggregate per-run FT-FSOD CDFSOD result JSONs into a mean +/- std table.

Usage: python3 aggregate_results.py <results_dir> [--out table.md]
"""
import argparse
import glob
import json
import os
import re
import statistics
from typing import Dict, List, Tuple

RUN_NAME_RE = re.compile(r"^swinB_(?P<domain>.+)_(?P<shot>\d+)shot_seed(?P<seed>\d+)$")


def load_results(results_dir: str) -> Dict[Tuple[str, str], List[float]]:
    """Group per-run mAP values by (domain, shot). Values are percentages."""
    results = {}  # type: Dict[Tuple[str, str], List[float]]
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        run_name = os.path.splitext(os.path.basename(path))[0]
        match = RUN_NAME_RE.match(run_name)
        if not match:
            continue
        with open(path) as f:
            data = json.load(f)
        map_value = data.get("coco/bbox_mAP")
        if map_value is None:
            continue
        key = (match.group("domain"), match.group("shot"))
        results.setdefault(key, []).append(round(map_value * 100, 4))
    return results


def summarize(results: Dict[Tuple[str, str], List[float]]) -> List[dict]:
    """Turn grouped results into sorted (domain, shot, mean, std, n) rows."""
    rows = []
    for (domain, shot), values in results.items():
        rows.append({
            "domain": domain,
            "shot": shot,
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        })
    rows.sort(key=lambda r: (r["domain"], int(r["shot"])))
    return rows


def render_markdown(rows: List[dict]) -> str:
    # Plain ASCII only: this machine's locale is "C", so non-ASCII output
    # (e.g. U+00B1 "+/-") crashes `print()` under the system Python 3.6.8
    # that runs this script.
    lines = ["| Domain | Shot | mAP (mean +/- std) | N seeds |",
             "|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['domain']} | {r['shot']} | {r['mean']:.2f} +/- {r['std']:.2f} | {r['n']} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results = load_results(args.results_dir)
    rows = summarize(results)
    table = render_markdown(rows)
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
