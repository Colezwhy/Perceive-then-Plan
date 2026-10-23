"""Evaluate Perceiver predictions.

    python tools/evaluate.py --predictions results/predictions.json --gt data/omni3d/eval_gt.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ptp_perceiver.box3d import is_vec3, local_axis_to_corners  # noqa: E402
from ptp_perceiver.prompt import parse_prediction  # noqa: E402


def generic_intrinsics(width, height):
    f = 2.0 * height / 2  # NDC focal length 2.0
    return np.array([[f, 0.0, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]])


def project_box(corners, K):
    pts = corners / corners[:, 2:3]
    pixels = (K @ pts.T).T[:, :2]
    x1, y1 = pixels.min(axis=0)
    x2, y2 = pixels.max(axis=0)
    return np.array([x1, y1, x2, y2])


def box_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6
    return inter / union


def precision_at(ious, thr):
    ious = np.asarray(ious)
    tp = (ious >= thr).sum()
    fp = (ious < thr).sum()
    return tp / (tp + fp + 1e-6)


def predicted_boxes(box_list):
    """Well-formed predicted boxes as (id, corners); malformed entries are dropped."""
    out = []
    for inst in box_list:
        if not isinstance(inst, dict):
            continue
        center, size = inst.get("center"), inst.get("size")
        x_axis, y_axis = inst.get("x_axis"), inst.get("y_axis")
        if not (is_vec3(center) and is_vec3(size) and is_vec3(x_axis) and is_vec3(y_axis)):
            continue
        # Orthonormalize the predicted axes, then build the corners.
        x = np.asarray(x_axis, dtype=np.float64)
        y = np.asarray(y_axis, dtype=np.float64)
        x_hat = x / np.linalg.norm(x)
        y = y - np.dot(y, x_hat) * x_hat
        y_hat = y / np.linalg.norm(y)
        out.append((inst.get("id"), local_axis_to_corners(center, size, x_hat, y_hat)))
    return out


def evaluate(predictions, gt_by_id, match="index", intrinsics="generic"):
    ious, depth_errors = [], []
    n_images = n_unparsed = 0
    for pred in predictions:
        gt = gt_by_id[pred["id"]]
        n_images += 1
        box_list = parse_prediction(pred["prediction"])
        if box_list is None:
            n_unparsed += 1
            continue
        W, H = gt["width"], gt["height"]
        K = generic_intrinsics(W, H) if intrinsics == "generic" else np.asarray(gt["K"], dtype=np.float64)
        objects = gt["objects"]
        objects_by_id = {o["id"]: o for o in objects}

        for num, (pred_id, corners) in enumerate(predicted_boxes(box_list)):
            if match == "index":
                if num >= len(objects):
                    continue
                obj = objects[num]
            else:
                obj = objects_by_id.get(pred_id)
                if obj is None:
                    continue
            if obj["bbox_3d"] is None or not any(obj["box_2d"]):
                continue
            b = obj["box_2d"]
            box_px = [b[0] / 1000.0 * W, b[1] / 1000.0 * H, b[2] / 1000.0 * W, b[3] / 1000.0 * H]
            ious.append(box_iou(project_box(corners, K), box_px))
            z_pred = corners.mean(axis=0)[2]
            z_gt = np.asarray(obj["bbox_3d"], dtype=np.float32).mean(axis=0)[2]
            depth_errors.append(abs(z_pred - z_gt) / (z_gt + 1e-6))

    return {
        "reproj_iou": float(np.mean(ious)),
        "prec@0.25": float(precision_at(ious, 0.25)),
        "prec@0.5": float(precision_at(ious, 0.5)),
        "depth_error": float(np.mean(depth_errors)),
        "num_images": n_images,
        "num_unparsed_images": n_unparsed,
        "num_scored_boxes": len(ious),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True, help="Output of tools/infer.py")
    p.add_argument("--gt", required=True, help="--gt-output of tools/prepare_omni3d.py --mode eval")
    p.add_argument("--match", choices=["index", "id"], default="index", help="Match predicted boxes to GT by list position or id")
    p.add_argument("--intrinsics", choices=["generic", "gt"], default="generic", help="Generic pinhole camera or Omni3D K")
    p.add_argument("--output", help="Optional JSON file for the metrics")
    args = p.parse_args()

    with open(args.predictions, "r") as f:
        predictions = json.load(f)
    with open(args.gt, "r") as f:
        gt_by_id = {g["id"]: g for g in json.load(f)}

    metrics = evaluate(predictions, gt_by_id, args.match, args.intrinsics)
    print(f"Reproj. IoU   {metrics['reproj_iou']:.4f}")
    print(f"Prec@0.25     {metrics['prec@0.25']:.4f}")
    print(f"Prec@0.5      {metrics['prec@0.5']:.4f}")
    print(f"Avg. DE (rel) {metrics['depth_error']:.4f}")
    print(f"({metrics['num_scored_boxes']} boxes from {metrics['num_images']} images, "
          f"{metrics['num_unparsed_images']} unparsable outputs)")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
