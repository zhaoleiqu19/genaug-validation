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
