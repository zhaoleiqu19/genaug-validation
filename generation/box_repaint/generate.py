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
    manifest_dir = os.path.dirname(args.manifest)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
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
    pipe.enable_group_offload(
        onload_device=torch.device("cuda:{}".format(args.gpu)),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=1,
        use_stream=True,
    )

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
