# clipart1k Box-Repaint Rung-1, Stage 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the generation pipeline, training integration, and results-comparison tooling for Stage 1 (1-shot only) of the clipart1k box-repaint rung-1 experiment, so a human can launch `experiments/clipart1k_box_repaint/run_stage1.sh` and get a decision-gated comparison against the frozen baselines.

**Architecture:** Two decoupled stages — a GPU-only `generate.py` (FLUX.1-dev box repaint, `flux2` env) that writes raw variant PNGs plus a CSV manifest, and a CPU-only `build_annotations.py` that merges the manifest with the official K-shot COCO json into a self-contained training set. Training reuses the existing `baselines/ftfsod_cdfsod/run_one.sh` via new optional environment-variable overrides (backward compatible with Phase-1's frozen baseline runs). A small `compare_results.py` computes the two spec'd readings (primary vs same-shot baseline, secondary vs next-tier baseline) and renders the decision-gate verdict.

**Tech Stack:** Python (repo-level `python3` = 3.6.8 for anything under `python3 -m pytest`; `flux2` conda env = PyTorch/diffusers/Pillow for generation; `ftfsod` conda env for MM Grounding DINO training), bash orchestration scripts, COCO-format JSON, pytest.

## Global Constraints

- Code imported by `python3 -m pytest` must stay Python 3.6.8-compatible and **must not import Pillow or torch** — the repo's system Python has neither installed. PIL-dependent code (`raster.py`) and the GPU script (`generate.py`) are never imported by test files; they run only inside the `flux2` conda env.
- Never write, edit, or delete anything under `/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/` (read-only shared data). Real support images are always *copied*, never modified in place.
- All generated/merged outputs go under `/data1/qushiduo/datasets/genaug/clipart1k/` (our own disk).
- Generator: FLUX.1-dev at `/data1/qushiduo/models/flux2/FLUX.1-dev`, `FluxInpaintPipeline`, strength fixed at `0.4`, `N=4` variants per annotation. Neither strength nor N is tuned against results in this plan — both are fixed inputs per `docs/specs/2026-07-08-clipart1k-box-repaint-rung1-design.md`.
- This plan covers **Stage 1 (1-shot) only**. Do not build or wire up 5-shot (Stage 2) generation/training — it is explicitly gated on Stage 1's results and out of scope here.
- `baselines/ftfsod_cdfsod/run_one.sh`'s existing behavior for its 5 original positional args must stay byte-for-byte unchanged (it reproduces the frozen Phase-1 baseline numbers) — new behavior is added only via optional environment variables that default to empty/unset.
- HF downloads: unset proxy env vars, `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` once the model is already cached locally (already the case — `FLUX.1-dev` is at `/data1/qushiduo/models/flux2/FLUX.1-dev` from the precheck).

---

## Task 1: Package skeleton + geometry.py (pure resize/scale math)

**Files:**
- Create: `generation/__init__.py`
- Create: `generation/box_repaint/__init__.py`
- Create: `generation/box_repaint/tests/__init__.py`
- Create: `generation/box_repaint/geometry.py`
- Test: `generation/box_repaint/tests/test_geometry.py`

**Interfaces:**
- Produces: `round16(x: float) -> int`, `compute_resize(orig_w: int, orig_h: int, target_w: int = 1024) -> Tuple[int, int, float, float]` (returns `new_w, new_h, scale_x, scale_y`), `scale_bbox(bbox: Tuple[float, float, float, float], scale_x: float, scale_y: float) -> Tuple[float, float, float, float]` (bbox is `x, y, w, h`). All three are used by `generate.py` in Task 6.

- [ ] **Step 1: Create empty package files**

```bash
mkdir -p generation/box_repaint/tests
touch generation/__init__.py generation/box_repaint/__init__.py generation/box_repaint/tests/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `generation/box_repaint/tests/test_geometry.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest generation/box_repaint/tests/test_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'generation.box_repaint.geometry'`

- [ ] **Step 4: Write the implementation**

Create `generation/box_repaint/geometry.py`:

```python
"""Pure box/resize math shared by the box-repaint generation pipeline.

Deliberately free of PIL/torch imports: this module is exercised by
`python3 -m pytest` under the repo's system Python 3.6.8, which has
neither installed. PIL-dependent operations live in raster.py instead.
"""


def round16(x):
    """Round to the nearest multiple of 16, minimum 16 (FLUX requires
    both dimensions to be multiples of 16)."""
    return max(16, int(round(x / 16)) * 16)


def compute_resize(orig_w, orig_h, target_w=1024):
    """Return (new_w, new_h, scale_x, scale_y) so the resized width is
    close to target_w and both dimensions are multiples of 16."""
    scale = target_w / orig_w
    new_w = round16(orig_w * scale)
    new_h = round16(orig_h * scale)
    return new_w, new_h, new_w / orig_w, new_h / orig_h


def scale_bbox(bbox, scale_x, scale_y):
    """bbox = (x, y, w, h) in original pixels -> scaled pixels."""
    x, y, w, h = bbox
    return (x * scale_x, y * scale_y, w * scale_x, h * scale_y)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest generation/box_repaint/tests/test_geometry.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add generation/__init__.py generation/box_repaint/__init__.py \
    generation/box_repaint/tests/__init__.py generation/box_repaint/geometry.py \
    generation/box_repaint/tests/test_geometry.py
git commit -m "feat: add box-repaint geometry helpers"
```

---

## Task 2: prompts.py (per-category prompt template)

**Files:**
- Create: `generation/box_repaint/prompts.py`
- Test: `generation/box_repaint/tests/test_prompts.py`

**Interfaces:**
- Produces: `prompt_for_category(category_name: str) -> str`, used by `generate.py` in Task 6.

- [ ] **Step 1: Write the failing test**

Create `generation/box_repaint/tests/test_prompts.py`:

```python
from generation.box_repaint.prompts import prompt_for_category


def test_prompt_names_the_category():
    assert prompt_for_category("bird") == (
        "a bird, flat-color cartoon clipart illustration, bold black outlines")


def test_prompt_handles_multiword_category_name():
    assert prompt_for_category("dining table") == (
        "a dining table, flat-color cartoon clipart illustration, "
        "bold black outlines")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest generation/box_repaint/tests/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `generation/box_repaint/prompts.py`:

```python
"""Per-category prompt template for clipart1k box repaint.

Names the class in the prompt (opposite of the parked background-repaint
route, which omitted it to avoid spawning extra instances) to anchor
identity under partial repaint at strength<1 — see
report/genaug-rung0-precheck.md, Route 3.
"""

CLIPART_TEMPLATE = "a {cat}, flat-color cartoon clipart illustration, bold black outlines"


def prompt_for_category(category_name):
    return CLIPART_TEMPLATE.format(cat=category_name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest generation/box_repaint/tests/test_prompts.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add generation/box_repaint/prompts.py generation/box_repaint/tests/test_prompts.py
git commit -m "feat: add box-repaint prompt template"
```

---

## Task 3: manifest.py (generation manifest CSV read/write)

**Files:**
- Create: `generation/box_repaint/manifest.py`
- Test: `generation/box_repaint/tests/test_manifest.py`

**Interfaces:**
- Produces: `FIELDNAMES: List[str]`, `append_row(path: str, row: dict) -> None`, `read_manifest(path: str) -> List[dict]` (numeric fields converted to `int`/`float`), `existing_keys(path: str) -> Set[Tuple[int, int, int]]` (set of `(source_image_id, target_annotation_id, variant_index)`). Used by `generate.py` (Task 6, writes) and `build_annotations.py` (Task 4, reads).

- [ ] **Step 1: Write the failing test**

Create `generation/box_repaint/tests/test_manifest.py`:

```python
import os

from generation.box_repaint.manifest import append_row, read_manifest, existing_keys


SAMPLE_ROW = {
    "source_image_id": 75, "source_file": "61545482.jpg",
    "target_annotation_id": 228, "category_id": 1, "category_name": "sheep",
    "bbox_x": 58.0, "bbox_y": 148.0, "bbox_w": 127.0, "bbox_h": 82.0,
    "variant_index": 0, "seed": 0, "strength": 0.4,
    "output_path": "/tmp/gen_75_228_v0.png",
}


def test_append_row_creates_file_with_header(tmp_path):
    path = str(tmp_path / "manifest.csv")
    append_row(path, SAMPLE_ROW)

    assert os.path.exists(path)
    rows = read_manifest(path)
    assert len(rows) == 1
    assert rows[0]["source_image_id"] == 75
    assert rows[0]["bbox_x"] == 58.0
    assert rows[0]["strength"] == 0.4
    assert rows[0]["output_path"] == "/tmp/gen_75_228_v0.png"


def test_append_row_twice_does_not_duplicate_header(tmp_path):
    path = str(tmp_path / "manifest.csv")
    append_row(path, SAMPLE_ROW)
    second_row = dict(SAMPLE_ROW, variant_index=1)
    append_row(path, second_row)

    rows = read_manifest(path)
    assert len(rows) == 2
    assert [r["variant_index"] for r in rows] == [0, 1]


def test_existing_keys_reflects_written_rows(tmp_path):
    path = str(tmp_path / "manifest.csv")
    assert existing_keys(path) == set()

    append_row(path, SAMPLE_ROW)
    assert existing_keys(path) == {(75, 228, 0)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest generation/box_repaint/tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `generation/box_repaint/manifest.py`:

```python
"""CSV manifest of generated box-repaint variants. Read by both
generate.py (to skip already-generated variants on resume) and
build_annotations.py (to know what to merge). Stdlib only — importable
under the repo's system Python 3.6.8 for pytest, unlike generate.py.
"""
import csv
import os

FIELDNAMES = [
    "source_image_id", "source_file", "target_annotation_id",
    "category_id", "category_name",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "variant_index", "seed", "strength", "output_path",
]

_INT_FIELDS = ("source_image_id", "target_annotation_id", "category_id",
               "variant_index", "seed")
_FLOAT_FIELDS = ("bbox_x", "bbox_y", "bbox_w", "bbox_h", "strength")


def append_row(path, row):
    """Append one row, writing the header first if the file doesn't exist yet."""
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_manifest(path):
    """Return all rows with numeric fields converted to int/float."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = dict(raw)
            for field in _INT_FIELDS:
                row[field] = int(row[field])
            for field in _FLOAT_FIELDS:
                row[field] = float(row[field])
            rows.append(row)
    return rows


def existing_keys(path):
    """Return the set of (source_image_id, target_annotation_id,
    variant_index) triples already recorded, for resumable generation."""
    return {
        (row["source_image_id"], row["target_annotation_id"], row["variant_index"])
        for row in read_manifest(path)
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest generation/box_repaint/tests/test_manifest.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add generation/box_repaint/manifest.py generation/box_repaint/tests/test_manifest.py
git commit -m "feat: add box-repaint generation manifest read/write"
```

---

## Task 4: build_annotations.py (merge logic + CLI)

**Files:**
- Create: `generation/box_repaint/build_annotations.py`
- Test: `generation/box_repaint/tests/test_build_annotations.py`

**Interfaces:**
- Consumes: `generation.box_repaint.manifest.read_manifest` (Task 3).
- Produces: `build_merged_annotations(shot_data: dict, manifest_rows: List[dict]) -> Tuple[dict, List[str]]` (pure; returns the merged COCO dict and the list of synthetic file names added), `materialize(shot_data: dict, manifest_rows: List[dict], real_images_dir: str, out_dir: str) -> dict` (I/O wrapper: copies images, writes `out_dir/annotations.json`, returns the merged dict). Used by `run_stage1.sh` (Task 9) via the CLI (`main()`).

- [ ] **Step 1: Write the failing tests**

Create `generation/box_repaint/tests/test_build_annotations.py`:

```python
import json

from generation.box_repaint.build_annotations import build_merged_annotations, materialize


def _shot_data_single_box():
    return {
        "images": [{"id": 75, "file_name": "61545482.jpg", "width": 350, "height": 244}],
        "annotations": [
            {"id": 228, "image_id": 75, "bbox": [58.0, 148.0, 127.0, 82.0],
             "area": 10414.0, "category_id": 1, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "sheep", "supercategory": "none"}],
    }


def test_single_box_image_gets_one_new_image_and_annotation_per_variant():
    shot_data = _shot_data_single_box()
    manifest_rows = [
        {"source_image_id": 75, "output_path": "/tmp/gen_75_228_v0.png"},
        {"source_image_id": 75, "output_path": "/tmp/gen_75_228_v1.png"},
    ]

    merged, synthetic_names = build_merged_annotations(shot_data, manifest_rows)

    assert len(merged["images"]) == 3
    assert len(merged["annotations"]) == 3
    assert synthetic_names == ["gen_75_228_v0.png", "gen_75_228_v1.png"]

    new_images = [img for img in merged["images"] if img["id"] != 75]
    assert {img["id"] for img in new_images} == {76, 77}
    for img in new_images:
        assert img["width"] == 350 and img["height"] == 244

    new_anns = [a for a in merged["annotations"] if a["id"] != 228]
    assert len(new_anns) == 2
    for ann in new_anns:
        assert ann["bbox"] == [58.0, 148.0, 127.0, 82.0]
        assert ann["category_id"] == 1
        assert ann["image_id"] in {76, 77}


def test_multi_box_image_carries_both_annotations_into_every_variant():
    shot_data = {
        "images": [{"id": 10, "file_name": "multi.jpg", "width": 500, "height": 400}],
        "annotations": [
            {"id": 1, "image_id": 10, "bbox": [0.0, 0.0, 50.0, 50.0],
             "area": 2500.0, "category_id": 1, "iscrowd": 0},
            {"id": 2, "image_id": 10, "bbox": [100.0, 100.0, 60.0, 60.0],
             "area": 3600.0, "category_id": 2, "iscrowd": 0},
        ],
        "categories": [
            {"id": 1, "name": "cat", "supercategory": "none"},
            {"id": 2, "name": "dog", "supercategory": "none"},
        ],
    }
    manifest_rows = [{"source_image_id": 10, "output_path": "/tmp/gen_10_1_v0.png"}]

    merged, _ = build_merged_annotations(shot_data, manifest_rows)

    assert len(merged["images"]) == 2
    new_image = next(img for img in merged["images"] if img["id"] != 10)
    new_anns = [a for a in merged["annotations"] if a["image_id"] == new_image["id"]]
    assert len(new_anns) == 2
    assert {a["category_id"] for a in new_anns} == {1, 2}
    assert {tuple(a["bbox"]) for a in new_anns} == {
        (0.0, 0.0, 50.0, 50.0), (100.0, 100.0, 60.0, 60.0)}


def test_new_annotation_ids_do_not_collide_with_real_ones():
    shot_data = _shot_data_single_box()
    manifest_rows = [{"source_image_id": 75, "output_path": "/tmp/x.png"}]

    merged, _ = build_merged_annotations(shot_data, manifest_rows)

    ids = [a["id"] for a in merged["annotations"]]
    assert len(ids) == len(set(ids))
    assert 228 in ids


def test_no_manifest_rows_returns_shot_data_unchanged():
    shot_data = _shot_data_single_box()
    merged, synthetic_names = build_merged_annotations(shot_data, [])
    assert merged["images"] == shot_data["images"]
    assert merged["annotations"] == shot_data["annotations"]
    assert synthetic_names == []


def test_materialize_copies_images_and_writes_annotations_json(tmp_path):
    shot_data = _shot_data_single_box()
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "61545482.jpg").write_bytes(b"fake-jpg-bytes")

    gen_file = tmp_path / "gen_75_228_v0.png"
    gen_file.write_bytes(b"fake-png-bytes")
    manifest_rows = [{"source_image_id": 75, "output_path": str(gen_file)}]

    out_dir = tmp_path / "out"
    merged = materialize(shot_data, manifest_rows, str(real_dir), str(out_dir))

    assert (out_dir / "images" / "61545482.jpg").read_bytes() == b"fake-jpg-bytes"
    assert (out_dir / "images" / "gen_75_228_v0.png").read_bytes() == b"fake-png-bytes"

    with open(str(out_dir / "annotations.json")) as f:
        on_disk = json.load(f)
    assert on_disk == merged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest generation/box_repaint/tests/test_build_annotations.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `generation/box_repaint/build_annotations.py`:

```python
"""Stage 2 (CPU-only): merge real support images with generated box-repaint
variants into one self-contained COCO-format training set.

Usage:
    python3 -m generation.box_repaint.build_annotations \
        --shot-json /data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/clipart1k/annotations/1_shot.json \
        --real-images-dir /data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/clipart1k/train \
        --manifest /data1/qushiduo/datasets/genaug/clipart1k/raw_generations/manifest_1shot.csv \
        --out-dir /data1/qushiduo/datasets/genaug/clipart1k/1shot
"""
import argparse
import json
import os
import shutil

from generation.box_repaint.manifest import read_manifest


def build_merged_annotations(shot_data, manifest_rows):
    """Pure merge: every real image/annotation is kept unchanged; each
    manifest row adds one new image (a box-repaint variant of its source
    image) carrying copies of ALL of that source image's annotations
    (coordinates unchanged — only pixels inside the targeted box differ,
    everything else, including any other GT box on the same image, is
    pixel-frozen by construction in generate.py).

    Returns (merged_coco_dict, list_of_synthetic_file_names_in_row_order).
    """
    images_by_id = {img["id"]: img for img in shot_data["images"]}
    anns_by_image = {}
    for ann in shot_data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    next_image_id = max(images_by_id) + 1 if images_by_id else 1
    next_ann_id = max((a["id"] for a in shot_data["annotations"]), default=0) + 1

    merged_images = list(shot_data["images"])
    merged_annotations = list(shot_data["annotations"])
    synthetic_file_names = []

    for row in manifest_rows:
        src_id = row["source_image_id"]
        src_img = images_by_id[src_id]
        new_image_id = next_image_id
        next_image_id += 1
        file_name = os.path.basename(row["output_path"])
        merged_images.append({
            "id": new_image_id,
            "file_name": file_name,
            "width": src_img["width"],
            "height": src_img["height"],
        })
        synthetic_file_names.append(file_name)
        for ann in anns_by_image.get(src_id, []):
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = new_image_id
            next_ann_id += 1
            merged_annotations.append(new_ann)

    merged = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": shot_data["categories"],
    }
    return merged, synthetic_file_names


def materialize(shot_data, manifest_rows, real_images_dir, out_dir):
    """I/O wrapper: copies real + synthetic images into out_dir/images/,
    writes out_dir/annotations.json. Returns the merged dict."""
    merged, _ = build_merged_annotations(shot_data, manifest_rows)

    images_out = os.path.join(out_dir, "images")
    os.makedirs(images_out, exist_ok=True)

    for img in shot_data["images"]:
        src = os.path.join(real_images_dir, img["file_name"])
        dst = os.path.join(images_out, img["file_name"])
        shutil.copyfile(src, dst)

    for row in manifest_rows:
        dst = os.path.join(images_out, os.path.basename(row["output_path"]))
        shutil.copyfile(row["output_path"], dst)

    ann_path = os.path.join(out_dir, "annotations.json")
    with open(ann_path, "w") as f:
        json.dump(merged, f)

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot-json", required=True)
    parser.add_argument("--real-images-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    with open(args.shot_json) as f:
        shot_data = json.load(f)
    manifest_rows = read_manifest(args.manifest)

    merged = materialize(shot_data, manifest_rows, args.real_images_dir, args.out_dir)
    print("[build_annotations] wrote {} images, {} annotations to {}".format(
        len(merged["images"]), len(merged["annotations"]), args.out_dir))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest generation/box_repaint/tests/test_build_annotations.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add generation/box_repaint/build_annotations.py generation/box_repaint/tests/test_build_annotations.py
git commit -m "feat: add box-repaint annotation merger"
```

---

## Task 5: raster.py (PIL-dependent mask/freeze operations — no pytest coverage)

**Files:**
- Create: `generation/box_repaint/raster.py`

**Interfaces:**
- Produces: `load_and_resize(path: str, new_w: int, new_h: int) -> PIL.Image`, `build_box_mask(size: Tuple[int, int], box: Tuple[float, float, float, float]) -> PIL.Image` ('L' mode, 255=repaint/inside box, 0=frozen), `freeze_outside_box(generated_img: PIL.Image, source_img: PIL.Image, box: Tuple[float, float, float, float]) -> PIL.Image`. Used by `generate.py` (Task 6) only — never imported by a test file, since Pillow is not installed under the repo's system Python 3.6.8 used for `python3 -m pytest`. Correctness is checked by the Task 10 manual smoke test instead.

- [ ] **Step 1: Write the implementation directly (no pytest step — see Interfaces note)**

Create `generation/box_repaint/raster.py`:

```python
"""PIL-dependent image operations for box repaint.

Not covered by `python3 -m pytest`: the repo's system Python 3.6.8 has no
Pillow installed. These functions only ever run inside the `flux2` conda
env via generate.py, and are checked by the manual smoke test in
docs/plans/2026-07-08-clipart1k-box-repaint-stage1.md Task 10, not by
unit tests.
"""
from PIL import Image, ImageDraw


def load_and_resize(path, new_w, new_h):
    img = Image.open(path).convert("RGB")
    return img.resize((new_w, new_h), Image.LANCZOS)


def build_box_mask(size, box):
    """size=(w,h); box=(x,y,w,h) in the same pixel space as size.
    Returns an 'L' mode mask: 255 = repaint (inside box), 0 = frozen."""
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    x, y, w, h = box
    d.rectangle([x, y, x + w, y + h], fill=255)
    return mask


def freeze_outside_box(generated_img, source_img, box):
    """Paste only the box region of generated_img onto a copy of
    source_img; every other pixel stays identical to source_img."""
    out = source_img.copy()
    x, y, w, h = box
    left, top = int(round(x)), int(round(y))
    right, bottom = int(round(x + w)), int(round(y + h))
    region = generated_img.crop((left, top, right, bottom))
    out.paste(region, (left, top))
    return out
```

- [ ] **Step 2: Verify it's syntactically valid Python** (cheapest check available without Pillow installed)

Run: `python3 -c "import ast; ast.parse(open('generation/box_repaint/raster.py').read())"`
Expected: no output, exit code 0

- [ ] **Step 3: Commit**

```bash
git add generation/box_repaint/raster.py
git commit -m "feat: add box-repaint raster operations"
```

---

## Task 6: generate.py (GPU stage CLI)

**Files:**
- Create: `generation/box_repaint/generate.py`

**Interfaces:**
- Consumes: `geometry.compute_resize`, `geometry.scale_bbox` (Task 1); `prompts.prompt_for_category` (Task 2); `manifest.append_row`, `manifest.existing_keys` (Task 3); `raster.load_and_resize`, `raster.build_box_mask`, `raster.freeze_outside_box` (Task 5).
- Produces: one PNG per (annotation, variant) under `--out-dir`, and manifest rows via `manifest.append_row`. Consumed by `build_annotations.py` (Task 4) and `run_stage1.sh` (Task 9).

Not unit-tested (requires a GPU, `torch`, `diffusers`, and the downloaded FLUX.1-dev weights, none of which are available under `python3 -m pytest`'s environment). Verified by the Task 10 manual smoke test.

- [ ] **Step 1: Write the implementation**

Create `generation/box_repaint/generate.py`:

```python
"""Stage 1 (GPU): generate box-repaint variants for one K-shot split.

Must run inside the `flux2` conda env (torch + diffusers + Pillow).
Resumable: re-running skips (source_image, annotation, variant) triples
already recorded in the manifest.

Usage:
    conda run -n flux2 python3 -m generation.box_repaint.generate \
        --shot-json /data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/clipart1k/annotations/1_shot.json \
        --images-dir /data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/clipart1k/train \
        --model-dir /data1/qushiduo/models/flux2/FLUX.1-dev \
        --out-dir /data1/qushiduo/datasets/genaug/clipart1k/raw_generations/1shot \
        --manifest /data1/qushiduo/datasets/genaug/clipart1k/raw_generations/manifest_1shot.csv \
        --n-variants 4 --strength 0.4 --gpu 0
"""
import os
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
           "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json

import torch
from PIL import Image
from diffusers import FluxInpaintPipeline

from generation.box_repaint.geometry import compute_resize, scale_bbox
from generation.box_repaint.manifest import append_row, existing_keys
from generation.box_repaint.prompts import prompt_for_category
from generation.box_repaint.raster import build_box_mask, freeze_outside_box, load_and_resize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot-json", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--n-variants", type=int, default=4)
    parser.add_argument("--strength", type=float, default=0.4)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N annotations (smoke testing)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.cuda.set_device(args.gpu)
    device = "cuda:{}".format(args.gpu)

    with open(args.shot_json) as f:
        shot_data = json.load(f)
    images_by_id = {img["id"]: img for img in shot_data["images"]}
    categories_by_id = {c["id"]: c["name"] for c in shot_data["categories"]}
    annotations = shot_data["annotations"]
    if args.limit:
        annotations = annotations[: args.limit]

    done = existing_keys(args.manifest)

    print("[load] {}".format(args.model_dir), flush=True)
    pipe = FluxInpaintPipeline.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload(gpu_id=args.gpu)

    for ann in annotations:
        image_id = ann["image_id"]
        annotation_id = ann["id"]
        category_id = ann["category_id"]
        category_name = categories_by_id[category_id]
        src_meta = images_by_id[image_id]
        src_path = os.path.join(args.images_dir, src_meta["file_name"])

        source_img = Image.open(src_path).convert("RGB")
        orig_w, orig_h = source_img.size
        new_w, new_h, sx, sy = compute_resize(orig_w, orig_h)
        resized_img = load_and_resize(src_path, new_w, new_h)
        box_r = scale_bbox(tuple(ann["bbox"]), sx, sy)
        mask = build_box_mask((new_w, new_h), box_r)
        prompt = prompt_for_category(category_name)

        for variant_index in range(args.n_variants):
            key = (image_id, annotation_id, variant_index)
            if key in done:
                print("[skip] ann={} v{} already in manifest".format(
                    annotation_id, variant_index), flush=True)
                continue
            seed = args.seed_base + variant_index
            file_name = "gen_{}_{}_v{}.png".format(image_id, annotation_id, variant_index)
            output_path = os.path.join(args.out_dir, file_name)

            generator = torch.Generator(device=device).manual_seed(seed)
            gen_out = pipe(
                prompt=prompt,
                image=resized_img,
                mask_image=mask,
                height=new_h,
                width=new_w,
                strength=args.strength,
                num_inference_steps=args.steps,
                guidance_scale=3.5,
                generator=generator,
            ).images[0]

            gen_out_orig_size = gen_out.resize((orig_w, orig_h), Image.LANCZOS)
            final_img = freeze_outside_box(gen_out_orig_size, source_img, tuple(ann["bbox"]))
            final_img.save(output_path)

            append_row(args.manifest, {
                "source_image_id": image_id,
                "source_file": src_meta["file_name"],
                "target_annotation_id": annotation_id,
                "category_id": category_id,
                "category_name": category_name,
                "bbox_x": ann["bbox"][0], "bbox_y": ann["bbox"][1],
                "bbox_w": ann["bbox"][2], "bbox_h": ann["bbox"][3],
                "variant_index": variant_index,
                "seed": seed,
                "strength": args.strength,
                "output_path": output_path,
            })
            print("[gen] {} ann={} v{} -> {}".format(
                src_meta["file_name"], annotation_id, variant_index, output_path), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it's syntactically valid Python**

Run: `python3 -c "import ast; ast.parse(open('generation/box_repaint/generate.py').read())"`
Expected: no output, exit code 0

- [ ] **Step 3: Commit**

```bash
git add generation/box_repaint/generate.py
git commit -m "feat: add box-repaint GPU generation script"
```

---

## Task 7: Extend run_one.sh with optional dataset-override env vars

**Files:**
- Modify: `baselines/ftfsod_cdfsod/run_one.sh`

**Interfaces:**
- Produces: three new optional env vars (`TRAIN_DATA_ROOT`, `TRAIN_ANN_FILE`, `TRAIN_IMG_PREFIX`) that, if set, add `--cfg-options` overrides pointing the training dataset at a different `data_root`/`ann_file`/`data_prefix.img`; `RUN_TAG` (appended to the run name, default empty) and `RESULTS_DIR` (default unchanged: `${SCRIPT_DIR}/results`) so augmented-run results never collide with the frozen baseline's result files. Consumed by `run_stage1.sh` (Task 9).

No unit test: this is a bash script with no Python entry point, consistent with the existing repo convention (`run_one.sh` itself has no test file today — only `aggregate_results.py`, its Python sibling, is tested). Verified by a syntax check and by the Task 10 manual smoke test.

- [ ] **Step 1: Read the current file to confirm line numbers before editing**

Run: `grep -n "RUN_NAME=\|RESULTS_DIR=\|dist_train.sh\|cfg-options" baselines/ftfsod_cdfsod/run_one.sh`

Expected output includes these lines (confirm exact text before editing, since line numbers may have drifted):
```
RESULTS_DIR="${SCRIPT_DIR}/results"
RUN_NAME="swinB_${DOMAIN}_${SHOT}shot_seed${SEED}"
conda run -n ftfsod ./tools/dist_train.sh "${CONFIG}" 1 "${PORT}" "${GPU}" \
    --work-dir "${TRAIN_WORK_DIR}" \
    --cfg-options randomness.seed="${SEED}"
```

- [ ] **Step 2: Edit `RESULTS_DIR` and `RUN_NAME` to be overridable**

Change:
```bash
FTFSOD_REPO="${FTFSOD_REPO:-$HOME/external/FT-FSOD}"
WORK_ROOT="${WORK_ROOT:-/data1/qushiduo/experiments/ftfsod_cdfsod/work_dirs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
```
to:
```bash
FTFSOD_REPO="${FTFSOD_REPO:-$HOME/external/FT-FSOD}"
WORK_ROOT="${WORK_ROOT:-/data1/qushiduo/experiments/ftfsod_cdfsod/work_dirs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
```

And change:
```bash
CONFIG="${FTFSOD_REPO}/configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_${DOMAIN}_${SHOT}shot.py"
RUN_NAME="swinB_${DOMAIN}_${SHOT}shot_seed${SEED}"
```
to:
```bash
CONFIG="${FTFSOD_REPO}/configs_cdfsod/final_configs_bs4/grounding_dino_swin-b_finetune_${DOMAIN}_${SHOT}shot.py"
RUN_NAME="swinB_${DOMAIN}_${SHOT}shot_seed${SEED}${RUN_TAG:-}"
```

- [ ] **Step 3: Add optional dataset-override cfg-options before the train invocation**

Change:
```bash
echo "[run_one] train: domain=${DOMAIN} shot=${SHOT} seed=${SEED} gpu=${GPU}"
START_TS=$(date +%s)
conda run -n ftfsod ./tools/dist_train.sh "${CONFIG}" 1 "${PORT}" "${GPU}" \
    --work-dir "${TRAIN_WORK_DIR}" \
    --cfg-options randomness.seed="${SEED}"
```
to:
```bash
echo "[run_one] train: domain=${DOMAIN} shot=${SHOT} seed=${SEED} gpu=${GPU}"
START_TS=$(date +%s)

# Optional dataset-path overrides (set by augmented-data experiments, e.g.
# experiments/clipart1k_box_repaint/run_stage1.sh). Unset by default, which
# preserves the exact behavior this frozen baseline was reproduced with.
CFG_OPTIONS=(randomness.seed="${SEED}")
if [ -n "${TRAIN_DATA_ROOT:-}" ]; then
  CFG_OPTIONS+=(train_dataloader.dataset.data_root="${TRAIN_DATA_ROOT}")
fi
if [ -n "${TRAIN_ANN_FILE:-}" ]; then
  CFG_OPTIONS+=(train_dataloader.dataset.ann_file="${TRAIN_ANN_FILE}")
fi
if [ -n "${TRAIN_IMG_PREFIX:-}" ]; then
  CFG_OPTIONS+=(train_dataloader.dataset.data_prefix.img="${TRAIN_IMG_PREFIX}")
fi

conda run -n ftfsod ./tools/dist_train.sh "${CONFIG}" 1 "${PORT}" "${GPU}" \
    --work-dir "${TRAIN_WORK_DIR}" \
    --cfg-options "${CFG_OPTIONS[@]}"
```

- [ ] **Step 4: Verify bash syntax and backward-compatible defaults**

Run: `bash -n baselines/ftfsod_cdfsod/run_one.sh`
Expected: no output, exit code 0

Run: `RUN_TAG= RESULTS_DIR= TRAIN_DATA_ROOT= bash -c 'source /dev/stdin <<< "RESULTS_DIR=\"\${RESULTS_DIR:-default}\"; RUN_NAME=\"name\${RUN_TAG:-}\"; echo \$RESULTS_DIR \$RUN_NAME"'`
Expected: `default name` — confirms unset/empty overrides fall back to the original defaults exactly (this is a quick inline check of the bash parameter-expansion idiom used, not a full script run, since a full run needs GPUs/conda envs not available for a syntax-only check)

- [ ] **Step 5: Commit**

```bash
git add baselines/ftfsod_cdfsod/run_one.sh
git commit -m "feat: allow run_one.sh to train on an overridden dataset path"
```

---

## Task 8: compare_results.py (Stage 1 primary + secondary readings)

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/clipart1k_box_repaint/__init__.py`
- Create: `experiments/clipart1k_box_repaint/tests/__init__.py`
- Create: `experiments/clipart1k_box_repaint/compare_results.py`
- Test: `experiments/clipart1k_box_repaint/tests/test_compare_results.py`

**Interfaces:**
- Produces: `load_cell(results_dir: str, domain: str, shot: str, tag: str = "") -> List[float]`, `summarize(values: List[float]) -> dict` (keys `mean`, `std`, `n`), `compare(baseline_values: List[float], augmented_values: List[float]) -> dict` (keys `baseline`, `augmented`, `delta`, `threshold`, `signal`), `render_report(primary: dict, secondary: dict) -> str`. Used by `run_stage1.sh` (Task 9).

- [ ] **Step 1: Create package files**

```bash
mkdir -p experiments/clipart1k_box_repaint/tests
touch experiments/__init__.py experiments/clipart1k_box_repaint/__init__.py \
    experiments/clipart1k_box_repaint/tests/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `experiments/clipart1k_box_repaint/tests/test_compare_results.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest experiments/clipart1k_box_repaint/tests/test_compare_results.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write the implementation**

Create `experiments/clipart1k_box_repaint/compare_results.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest experiments/clipart1k_box_repaint/tests/test_compare_results.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add experiments/__init__.py experiments/clipart1k_box_repaint/__init__.py \
    experiments/clipart1k_box_repaint/tests/__init__.py \
    experiments/clipart1k_box_repaint/compare_results.py \
    experiments/clipart1k_box_repaint/tests/test_compare_results.py
git commit -m "feat: add Stage 1 primary/secondary comparison report"
```

---

## Task 9: run_stage1.sh (orchestration launch script)

**Files:**
- Create: `experiments/clipart1k_box_repaint/run_stage1.sh`

**Interfaces:**
- Consumes: `generation/box_repaint/generate.py` (Task 6), `generation/box_repaint/build_annotations.py` (Task 4), `baselines/ftfsod_cdfsod/run_one.sh` (Task 7, via the new env vars), `experiments/clipart1k_box_repaint/compare_results.py` (Task 8).
- Produces: the full Stage 1 pipeline in one command; not invoked by anything else — this is the top-level entry point a human runs.

No unit test (orchestration shell script, no Python entry point — same rationale as `run_one.sh`). Verified by the Task 10 manual smoke test, which exercises its generate → build_annotations half at `--limit 1` scale.

- [ ] **Step 1: Write the script**

Create `experiments/clipart1k_box_repaint/run_stage1.sh`:

```bash
#!/usr/bin/env bash
# Stage 1 (1-shot only) launch script for the clipart1k box-repaint
# rung-1 experiment. See docs/specs/2026-07-08-clipart1k-box-repaint-rung1-design.md.
#
# Usage: experiments/clipart1k_box_repaint/run_stage1.sh <gpu_id> <port>
set -euo pipefail

GPU="$1"
PORT="$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLIPART_SRC="/data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/clipart1k"
GENAUG_ROOT="/data1/qushiduo/datasets/genaug/clipart1k"
RAW_DIR="${GENAUG_ROOT}/raw_generations/1shot"
MANIFEST="${GENAUG_ROOT}/raw_generations/manifest_1shot.csv"
MERGED_DIR="${GENAUG_ROOT}/1shot"
RESULTS_DIR="${REPO_ROOT}/experiments/clipart1k_box_repaint/results"

# `python3 -m generation.box_repaint.generate` etc. resolve their package
# imports off the current working directory — must run from repo root.
cd "${REPO_ROOT}"

mkdir -p "${RAW_DIR}" "${RESULTS_DIR}"

echo "[stage1] generating box-repaint variants (N=4, strength=0.4)"
conda run -n flux2 python3 -m generation.box_repaint.generate \
    --shot-json "${CLIPART_SRC}/annotations/1_shot.json" \
    --images-dir "${CLIPART_SRC}/train" \
    --model-dir /data1/qushiduo/models/flux2/FLUX.1-dev \
    --out-dir "${RAW_DIR}" \
    --manifest "${MANIFEST}" \
    --n-variants 4 --strength 0.4 --gpu "${GPU}"

echo "[stage1] building merged annotations"
python3 -m generation.box_repaint.build_annotations \
    --shot-json "${CLIPART_SRC}/annotations/1_shot.json" \
    --real-images-dir "${CLIPART_SRC}/train" \
    --manifest "${MANIFEST}" \
    --out-dir "${MERGED_DIR}"

echo "[stage1] training 3 seeds (42, 43, 44)"
for SEED in 42 43 44; do
    TRAIN_DATA_ROOT="${MERGED_DIR}" \
    TRAIN_ANN_FILE="annotations.json" \
    TRAIN_IMG_PREFIX="images/" \
    RUN_TAG="_boxrepaint" \
    RESULTS_DIR="${RESULTS_DIR}" \
    "${REPO_ROOT}/baselines/ftfsod_cdfsod/run_one.sh" clipart1k 1 "${SEED}" "${GPU}" "${PORT}"
done

echo "[stage1] comparing against frozen baselines"
python3 -m experiments.clipart1k_box_repaint.compare_results \
    --baseline-dir "${REPO_ROOT}/baselines/ftfsod_cdfsod/results" \
    --augmented-dir "${RESULTS_DIR}" \
    --out "${RESULTS_DIR}/stage1_report.md"
```

- [ ] **Step 2: Make it executable and verify syntax**

Run: `chmod +x experiments/clipart1k_box_repaint/run_stage1.sh && bash -n experiments/clipart1k_box_repaint/run_stage1.sh`
Expected: no output, exit code 0

- [ ] **Step 3: Commit**

```bash
git add experiments/clipart1k_box_repaint/run_stage1.sh
git commit -m "feat: add Stage 1 orchestration launch script"
```

---

## Task 10: Manual smoke test (integration checkpoint before a full Stage 1 run)

This task is a documented manual verification, not a coded deliverable — it's the one integration point that needs a real GPU and cannot be exercised by `python3 -m pytest`. It catches wiring bugs (wrong paths, PIL/torch import errors, box-mask mistakes) cheaply, at `--limit 1` scale (one annotation, ≈45s of GPU time), before committing to the full 80-image / 3-seed Stage 1 run.

**Files:** none created — this task runs the code from Tasks 4–6 against real data.

All commands below assume the current working directory is the repo root (`cd ~/projects/genaug-validation`) — the `-m generation.box_repaint...` module invocations resolve their package imports off the working directory.

- [ ] **Step 1: Run generate.py against exactly one annotation**

```bash
conda run -n flux2 python3 -m generation.box_repaint.generate \
    --shot-json /data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/clipart1k/annotations/1_shot.json \
    --images-dir /data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/clipart1k/train \
    --model-dir /data1/qushiduo/models/flux2/FLUX.1-dev \
    --out-dir /tmp/box_repaint_smoke/raw \
    --manifest /tmp/box_repaint_smoke/manifest.csv \
    --n-variants 1 --strength 0.4 --gpu 0 --limit 1
```

Expected: prints `[load] ...`, then one `[gen] ... -> /tmp/box_repaint_smoke/raw/gen_<id>_<id>_v0.png` line, exits 0. Takes roughly 45–60 seconds (model load + one denoise pass, per the precheck's measured per-image timing).

- [ ] **Step 2: Inspect the manifest**

Run: `cat /tmp/box_repaint_smoke/manifest.csv`
Expected: a header row plus exactly one data row, with `variant_index=0`, `strength=0.4`, and an `output_path` matching the file printed in Step 1.

- [ ] **Step 3: Visually confirm the output image looks right**

Open `/tmp/box_repaint_smoke/raw/gen_<id>_<id>_v0.png` (e.g. via the Read tool, or `scp`/copy it locally) side by side with the source image. Confirm: background pixels outside the annotated box are unchanged from the source, the object inside the box is still recognizably the same class, and there's no visible seam at the box boundary. This is the same visual bar the precheck already validated at strength 0.4 (6/6 clean) — the goal here is only to catch a *pipeline wiring* regression (wrong box coordinates, wrong crop), not to re-validate the method itself.

- [ ] **Step 4: Run build_annotations.py on the one-annotation output**

```bash
python3 -m generation.box_repaint.build_annotations \
    --shot-json /data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/clipart1k/annotations/1_shot.json \
    --real-images-dir /data6022/xuanlong/datasets/NTIRE2025_CDFSOD/datasets/clipart1k/train \
    --manifest /tmp/box_repaint_smoke/manifest.csv \
    --out-dir /tmp/box_repaint_smoke/merged
```

Expected: prints `[build_annotations] wrote 21 images, 21 annotations to /tmp/box_repaint_smoke/merged` (20 real images/annotations from the full 1-shot split + 1 synthetic).

- [ ] **Step 5: Verify the merged json structure**

```bash
python3 -c "
import json
d = json.load(open('/tmp/box_repaint_smoke/merged/annotations.json'))
assert len(d['images']) == 21
assert len(d['annotations']) == 21
assert len(d['categories']) == 20
synth = [img for img in d['images'] if img['file_name'].startswith('gen_')]
assert len(synth) == 1
print('OK:', synth[0])
"
```
Expected: `OK: {'id': ..., 'file_name': 'gen_..._v0.png', 'width': ..., 'height': ...}`, exit code 0.

- [ ] **Step 6: Clean up the smoke-test scratch directory**

Run: `rm -rf /tmp/box_repaint_smoke`

- [ ] **Step 7: Report results in this session** (no commit — nothing new to commit from this task; it only exercises code already committed in Tasks 4–6)

If any step's expected output didn't match, fix the relevant task's code, re-run the earlier task's tests, and repeat this smoke test from Step 1 before considering the plan done.

---

## After this plan

This plan delivers the Stage 1 **tooling**, verified at `--limit 1` scale. It does **not** run the full Stage 1 experiment (80-image generation + 3-seed training, ≈1 GPU-hour generation + training time) — that's a deliberate, separate action: run `experiments/clipart1k_box_repaint/run_stage1.sh <gpu_id> <port>` when ready to spend that compute, then read `experiments/clipart1k_box_repaint/results/stage1_report.md` for the decision-gate verdict. Stage 2 (5-shot) is out of scope here entirely — per the spec, it only gets planned if Stage 1's primary reading clears the gate.
