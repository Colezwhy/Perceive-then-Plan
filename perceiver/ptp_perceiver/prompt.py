"""Perceiver prompt format and output parsing."""

import ast
import json
import re
from typing import List, Optional, Sequence, Tuple

PROMPT_HEADER = (
    "<image>\nHere are the detected 2D bounding boxes in this image, format: (id, class, x1, y1, x2, y2):"
)

OUTPUT_SCHEMA = (
    " Output a json list, where each entry is a 3D bounding box in the CAMERA coordinate system that "
    "corresponds to a given 2D bounding box. Each 3D bounding box must follow exactly this schema:\n"
    ' {"id": int,\n"class": str,\n"center": [position_u, position_v, position_x],\n'
    '"size":[scale_x, scale_y, scale_z],\n"x_axis": [x1, x2, x3],\n"y_axis": [y1, y2, y3]}\n\n'
    "Rules: The coordinate system is CAMERA: +u right, +v down, +x away from camera.\n"
    '"center" is the 3D box center in the GLOBAL camera coordinates.\n'
    '"size" is the length of the three edges along the box axes.\n'
    '"x_axis" and "y_axis" are unit vectors defining the LOCAL object frame of the box, must be orthogonal.\n'
    "Do NOT include explanations or extra text.\n"
    "All numbers must be valid floating point values."
)

_BOX_ROW = re.compile(
    r"\((\d+),\s*([^,]+?),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)"
)


def format_prompt(objects: Sequence[Tuple[int, str, Sequence[float]]]) -> str:
    """objects: (id, class name, [x1, y1, x2, y2] in Qwen 0-1000 coordinates), ids starting at 1.

    An object without a visible 2D box is written as [0, 0, 0, 0].
    """
    rows = [f"({i}, {cls}, {b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f}, {b[3]:.4f})" for i, cls, b in objects]
    return PROMPT_HEADER + " " + "\n".join(rows) + "\n\n" + OUTPUT_SCHEMA


def parse_prompt_boxes(prompt: str) -> List[Tuple[int, str, List[float]]]:
    """Inverse of `format_prompt`: [(id, class, [x1, y1, x2, y2])]."""
    return [
        (int(m[0]), m[1].strip(), [float(m[2]), float(m[3]), float(m[4]), float(m[5])])
        for m in _BOX_ROW.findall(prompt)
    ]


def parse_prediction(text) -> Optional[list]:
    """Parse the Perceiver output (Python or JSON list) into a list of box dicts, or None."""
    if isinstance(text, list):
        return text
    try:
        out = ast.literal_eval(text)
    except Exception:
        try:
            out = json.loads(text.strip().lstrip("```json").rstrip("```").strip())
        except Exception:
            return None
    return out if isinstance(out, list) else None
