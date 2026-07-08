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
