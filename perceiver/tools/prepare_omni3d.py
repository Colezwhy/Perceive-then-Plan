"""Build Perceiver training / evaluation data from Omni3D.

    python tools/prepare_omni3d.py --mode train --omni3d-json-dir $OMNI3D/datasets/Omni3D --output data/omni3d/train.json
    python tools/prepare_omni3d.py --mode eval --omni3d-json-dir $OMNI3D/datasets/Omni3D \
        --output data/omni3d/eval.json --gt-output data/omni3d/eval_gt.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ptp_perceiver.box3d import corners_to_local_axis, edge_lengths  # noqa: E402
from ptp_perceiver.prompt import format_prompt  # noqa: E402

DATASETS = ("SUNRGBD", "Hypersim")


def to_qwen_box(box, width, height):
    """Pixel [x1, y1, x2, y2] -> integer 0-1000 coordinates, clamped to the image. None box -> zeros."""
    if box is None or box[0] == -1:
        return [0, 0, 0, 0]
    q = [round(box[0] / width * 1000), round(box[1] / height * 1000),
         round(box[2] / width * 1000), round(box[3] / height * 1000)]
    return [min(1000, max(0, v)) for v in q]


def load_split(json_dir, dataset, split):
    with open(os.path.join(json_dir, f"{dataset}_{split}.json"), "r") as f:
        data = json.load(f)
    anns = defaultdict(list)
    for a in data["annotations"]:
        anns[a["image_id"]].append(a)
    return data["images"], anns


def select_train_objects(anns, args):
    selected, n_chairs = [], 0
    for a in anns:
        if not a["valid3D"] or a["behind_camera"] or a["category_name"] in args.exclude_categories:
            continue
        if a["visibility"] < 0 or a["visibility"] <= args.min_visibility:
            continue
        if a["truncation"] >= 0 and a["truncation"] >= args.max_truncation:
            continue
        if edge_lengths(a["bbox3D_cam"]).min() < args.min_edge:
            continue
        if a["category_name"] == "chair":
            if n_chairs >= args.max_chairs:
                continue
            n_chairs += 1
        selected.append(a)
        if len(selected) >= args.max_objects_per_image:
            break
    return selected


def build_train(args):
    records, stats = [], Counter()
    for dataset in args.datasets:
        for split in args.splits:
            images, anns_by_image = load_split(args.omni3d_json_dir, dataset, split)
            for im in images:
                stats["images"] += 1
                objects = select_train_objects(anns_by_image[im["id"]], args)
                if not (args.min_objects <= len(objects) <= args.max_objects):
                    stats["skipped_object_count"] += 1
                    continue
                W, H = im["width"], im["height"]
                prompt_objects, targets = [], []
                for k, a in enumerate(objects, start=1):
                    prompt_objects.append((k, a["category_name"], to_qwen_box(a["bbox2D_tight"], W, H)))
                    targets.append({"id": k, "class": a["category_name"], **corners_to_local_axis(a["bbox3D_cam"])})
                records.append({
                    "id": len(records),
                    "image": im["file_path"],
                    "conversations": [
                        {"from": "human", "value": format_prompt(prompt_objects)},
                        {"from": "gpt", "value": targets},
                    ],
                })
                stats[f"{dataset}_{split}"] += 1
                stats["objects"] += len(objects)
    return records, None, stats


def build_eval(args):
    prompts, gts, stats = [], [], Counter()
    for dataset in args.datasets:
        for split in args.splits:
            images, anns_by_image = load_split(args.omni3d_json_dir, dataset, split)
            for im in images:
                W, H = im["width"], im["height"]
                anns = [a for a in anns_by_image[im["id"]] if a["category_name"] not in args.exclude_categories]
                prompt_objects, gt_objects = [], []
                for k, a in enumerate(anns, start=1):
                    box = to_qwen_box(a["bbox2D_tight"], W, H)
                    prompt_objects.append((k, a["category_name"], box))
                    gt_objects.append({
                        "id": k,
                        "class": a["category_name"],
                        "box_2d": box,
                        "bbox_3d": a["bbox3D_cam"] if a["valid3D"] else None,
                    })
                    stats["scored_objects"] += int(a["valid3D"] and box != [0, 0, 0, 0])
                idx = len(prompts)
                prompts.append({
                    "id": idx,
                    "image": im["file_path"],
                    "messages": [{"role": "user", "content": format_prompt(prompt_objects)}],
                })
                gts.append({"id": idx, "image": im["file_path"], "dataset": dataset,
                            "width": W, "height": H, "K": im["K"], "objects": gt_objects})
                stats[f"{dataset}_{split}"] += 1
    return prompts, gts, stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["train", "eval"], required=True)
    p.add_argument("--omni3d-json-dir", required=True, help="Directory with Omni3D {Dataset}_{split}.json files")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--splits", nargs="+", default=None, help="Default: train val (train mode), test (eval mode)")
    p.add_argument("--output", required=True)
    p.add_argument("--gt-output", help="Ground-truth file (--mode eval)")
    p.add_argument("--exclude-categories", nargs="*", default=["window"])
    # --mode train only
    p.add_argument("--min-visibility", type=float, default=0.4)
    p.add_argument("--max-truncation", type=float, default=0.6)
    p.add_argument("--min-edge", type=float, default=1e-3, help="Skip boxes with an edge shorter than this (m)")
    p.add_argument("--max-chairs", type=int, default=6)
    p.add_argument("--max-objects-per-image", type=int, default=20)
    p.add_argument("--min-objects", type=int, default=1, help="Skip images with fewer selected objects")
    p.add_argument("--max-objects", type=int, default=1000, help="Skip images with more selected objects")
    args = p.parse_args()

    if args.splits is None:
        args.splits = ["train", "val"] if args.mode == "train" else ["test"]

    if args.mode == "train":
        records, gts, stats = build_train(args)
    else:
        if not args.gt_output:
            p.error("--gt-output is required with --mode eval")
        records, gts, stats = build_eval(args)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f)
    if gts is not None:
        with open(args.gt_output, "w") as f:
            json.dump(gts, f)

    print(f"Wrote {len(records)} records to {args.output}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
