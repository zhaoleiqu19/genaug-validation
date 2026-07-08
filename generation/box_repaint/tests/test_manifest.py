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
