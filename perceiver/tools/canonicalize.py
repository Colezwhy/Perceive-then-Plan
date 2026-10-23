"""Convert Perceiver predictions into grid layouts for Stage 2.

    python tools/canonicalize.py --predictions results/predictions.json --eval-json data/omni3d/eval.json \
        --output results/grid_layouts.json
"""

import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ptp_perceiver.box3d import is_vec3, local_axis_to_corners  # noqa: E402
from ptp_perceiver.prompt import parse_prediction, parse_prompt_boxes  # noqa: E402


def recover_rotation(x_axis, y_axis):
    """Gram-Schmidt re-orthogonalization of the predicted axes."""
    x = np.asarray(x_axis, dtype=np.float64)
    y = np.asarray(y_axis, dtype=np.float64)
    x_hat = x / (np.linalg.norm(x) + 1e-8)
    y = y - np.dot(y, x_hat) * x_hat
    y_hat = y / (np.linalg.norm(y) + 1e-8)
    return x_hat, y_hat


def parse_predicted_objects(box_list):
    """Predicted boxes -> dicts with camera-frame corners (Omni3D frame: X right, Y down, Z forward)."""
    parsed = []
    for inst in box_list:
        if not isinstance(inst, dict):
            continue
        center, size = inst.get("center"), inst.get("size")
        x_axis, y_axis = inst.get("x_axis"), inst.get("y_axis")
        if not (is_vec3(center) and is_vec3(size) and is_vec3(x_axis) and is_vec3(y_axis)):
            continue
        if all(s < 1e-6 for s in size):
            continue
        x_hat, y_hat = recover_rotation(x_axis, y_axis)
        parsed.append({
            "obj_id": inst.get("id"),
            "class": inst.get("class"),
            "vertices": local_axis_to_corners(center, size, x_hat, y_hat),
            "center_cam": list(center),
            "size_cam": list(size),
            "x_axis": x_hat.tolist(),
            "y_axis": y_hat.tolist(),
        })
    return parsed


# ------------------------------------------------------------------ gravity-aligned normalization


def estimate_gravity_robust(boxes, fallback_up=np.array([0.0, 1.0, 0.0])):
    """Up direction from the bottom-face normals of all boxes (Y-up convention)."""
    normals = []
    for box in boxes:
        n = np.cross(box[2] - box[3], box[7] - box[3])
        norm = np.linalg.norm(n)
        if norm > 1e-6:
            normals.append(n / norm)
    if len(normals) < 2:
        return fallback_up / np.linalg.norm(fallback_up), 0.0
    normals = np.array(normals)
    _, S, Vt = np.linalg.svd(normals)
    dominant = Vt[0]
    consistent = np.mean(np.abs(normals @ dominant) > 0.8)
    singularity_ratio = S[0] / S[1] if len(S) >= 2 and S[1] > 1e-6 else float("inf")
    confidence = min(1.0, consistent * min(singularity_ratio / 3.0, 1.0))
    if dominant[1] < 0:
        dominant = -dominant
    if confidence < 0.3:
        alpha = confidence / 0.3
        blended = alpha * dominant + (1 - alpha) * fallback_up
        return blended / np.linalg.norm(blended), confidence
    return dominant, confidence


def normalize_layout(boxes_raw):
    """Rotate boxes into a gravity-aligned, right-handed frame (Y up, X right, Z forward) and center them."""
    camera_forward = np.array([0.0, 0.0, 1.0])
    camera_right = np.array([1.0, 0.0, 0.0])

    # Y-down camera frame -> Y-up.
    y_max_global = np.max(np.concatenate(boxes_raw, axis=0)[:, 1])
    boxes = []
    for box in boxes_raw:
        b = box.copy()
        b[:, 1] = y_max_global - b[:, 1]
        boxes.append(b)

    up_cam, _ = estimate_gravity_robust(boxes)
    if up_cam[1] < 0:
        up_cam = -up_cam
    up_cam = up_cam / (np.linalg.norm(up_cam) + 1e-8)

    # x axis: camera right projected onto the ground plane, pointing to image right.
    x_axis = camera_right - np.dot(camera_right, up_cam) * up_cam
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        x_axis = camera_forward - np.dot(camera_forward, up_cam) * up_cam
        x_norm = np.linalg.norm(x_axis)
    x_axis = x_axis / (x_norm + 1e-8)
    if np.dot(x_axis, camera_right) < 0:
        x_axis = -x_axis

    z_axis = np.cross(x_axis, up_cam)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
    up_cam = np.cross(z_axis, x_axis)
    up_cam = up_cam / (np.linalg.norm(up_cam) + 1e-8)

    R_ga = np.stack([x_axis, up_cam, z_axis], axis=0)
    if np.linalg.det(R_ga) < 0:
        R_ga[2] *= -1

    aligned = [(R_ga @ box.T).T for box in boxes]
    all_points = np.concatenate(aligned, axis=0)
    origin_ga = np.array([
        (np.min(all_points[:, 0]) + np.max(all_points[:, 0])) / 2,
        np.min(all_points[:, 1]),
        (np.min(all_points[:, 2]) + np.max(all_points[:, 2])) / 2,
    ])
    return [box - origin_ga for box in aligned], R_ga, origin_ga, y_max_global


# ------------------------------------------------------------------ structured layout and grid


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def vertices_to_bbox(vertices):
    """Normalized corners -> (x, y_center, z, w, h, l, yaw)."""
    V = np.asarray(vertices)
    center = V.mean(axis=0)
    ex, ey, ez = V[1] - V[0], V[3] - V[0], V[4] - V[0]
    edge = ex.copy()
    edge[1] = 0
    if np.linalg.norm(edge) < 1e-6:
        yaw = 0.0
    else:
        edge = edge / np.linalg.norm(edge)
        yaw = np.arctan2(-edge[2], edge[0])
    return (center[0], center[1], center[2], np.linalg.norm(ex), np.linalg.norm(ey), np.linalg.norm(ez),
            normalize_angle(yaw))


def compute_pos_offset(instances, cell_size=0.10):
    layouts = np.array([ins["layout_3d"] for ins in instances])
    return np.min(layouts[:, 0:3] - layouts[:, 3:6] / 2.0, axis=0) - cell_size


def discretize_layout(layout_3d, pos_offset, cell_size=0.10, max_grid=256):
    x, y, z, w, h, l, yaw = layout_3d

    def disc(val, offset):
        return max(0, min(int(round((val - offset) / cell_size)), max_grid))

    yaw_norm = ((yaw + np.pi) % (2 * np.pi)) - np.pi
    gyaw = max(0, min(int(round(((yaw_norm + np.pi) / (2 * np.pi)) * 24)), 24))
    return [disc(x, pos_offset[0]), disc(y, pos_offset[1]), disc(z, pos_offset[2]),
            max(1, disc(w, 0)), max(1, disc(h, 0)), max(1, disc(l, 0)), gyaw]


def format_grid_layout_text(obj_id, cls, grid_layout, box_2d=None):
    """One object of the Planner's scene description; gy is shown at the bottom of the box."""
    gx, gy, gz, gw, gh, gl, gyaw = grid_layout
    lines = [f"obj_{obj_id} {cls}"]
    if box_2d is not None:
        lines.append(f"  bbox: [{box_2d[0]}, {box_2d[1]}, {box_2d[2]}, {box_2d[3]}]")
    lines.append(f"  pos: [{gx}, {max(0, gy - gh // 2)}, {gz}]")
    lines.append(f"  size: [{gw}, {gh}, {gl}]")
    lines.append(f"  yaw: {gyaw}")
    return "\n".join(lines)


def process_image(box_list, prompt, cell_size=0.10):
    """Grid layout for one image, or None if fewer than two objects have a visible 2D box."""
    boxes_2d = {i: b for i, _, b in parse_prompt_boxes(prompt)}
    parsed = parse_predicted_objects(box_list)
    for ins in parsed:
        raw = boxes_2d.get(ins["obj_id"])
        ins["box_2d"] = None
        if raw is not None:
            x1, y1, x2, y2 = raw
            if x2 > x1 and y2 > y1 and (x2 - x1) * (y2 - y1) > 1:
                ins["box_2d"] = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
    parsed = [ins for ins in parsed if ins["box_2d"] is not None]
    if len(parsed) < 2:
        return None

    try:
        normalized, R_ga, origin_ga, y_max_global = normalize_layout([ins["vertices"] for ins in parsed])
    except Exception as e:
        print(f"[WARN] Normalization failed: {e}")
        return None

    for ins, verts in zip(parsed, normalized):
        ins["layout_3d"] = [float(v) for v in vertices_to_bbox(verts)]
        ins["box_3d_corners_normalized"] = verts.tolist()
    pos_offset = compute_pos_offset(parsed, cell_size)
    for new_id, ins in enumerate(parsed):
        ins["layout_3d_grid"] = discretize_layout(ins["layout_3d"], pos_offset, cell_size)
        ins["obj_id"] = new_id

    lines = ["## Scene Layout (grid-based, 1 unit = 10cm)"]
    lines += [format_grid_layout_text(ins["obj_id"], ins["class"], ins["layout_3d_grid"], ins["box_2d"]) for ins in parsed]

    return {
        "R_ga": R_ga.tolist(),
        "origin_ga": origin_ga.tolist(),
        "y_max_global": float(y_max_global),
        "pos_offset": pos_offset.tolist(),
        "cell_size": cell_size,
        "instances": [{
            "obj_id": ins["obj_id"],
            "class": ins["class"],
            "layout_3d": ins["layout_3d"],
            "layout_3d_grid": ins["layout_3d_grid"],
            "box_2d": ins["box_2d"],
            "box_3d_corners_normalized": ins["box_3d_corners_normalized"],
            "center_cam_original": ins["center_cam"],
            "size_cam_original": ins["size_cam"],
            "x_axis_original": ins["x_axis"],
            "y_axis_original": ins["y_axis"],
        } for ins in parsed],
        "scene_layout": "\n".join(lines),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True, help="Output of tools/infer.py")
    p.add_argument("--eval-json", required=True, help="Prompts the predictions were made from (for 2D boxes)")
    p.add_argument("--output", required=True)
    p.add_argument("--cell-size", type=float, default=0.10, help="Grid cell size in meters")
    args = p.parse_args()

    with open(args.predictions, "r") as f:
        predictions = json.load(f)
    with open(args.eval_json, "r") as f:
        prompts = {r["id"]: r["messages"][0]["content"] for r in json.load(f)}

    layouts, skipped = [], 0
    for pred in tqdm(predictions, desc="Canonicalizing"):
        box_list = parse_prediction(pred["prediction"])
        layout = process_image(box_list, prompts[pred["id"]], args.cell_size) if box_list is not None else None
        if layout is None:
            skipped += 1
            continue
        layouts.append({"img_id": pred["id"], "image": pred["image"], **layout})

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(layouts, f)
    print(f"Wrote {len(layouts)} layouts to {args.output} ({skipped} images skipped)")


if __name__ == "__main__":
    main()
