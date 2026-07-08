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
