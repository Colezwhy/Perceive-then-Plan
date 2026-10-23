"""Local-axis 3D boxes <-> 8 corners in Omni3D order (camera frame: +x right, +y down, +z forward).

size = (|v1-v0|, |v3-v0|, |v4-v0|), x_axis along v1-v0, y_axis along v3-v0.
"""

from typing import Dict, Sequence

import numpy as np


def corners_to_local_axis(corners: Sequence[Sequence[float]]) -> Dict[str, list]:
    """8 Omni3D corners -> {"center", "size", "x_axis", "y_axis"}."""
    v = np.asarray(corners, dtype=np.float64)
    ex, ey, ez = v[1] - v[0], v[3] - v[0], v[4] - v[0]
    x = ex / np.linalg.norm(ex)
    y = ey - np.dot(ey, x) * x
    y = y / np.linalg.norm(y)
    return {
        "center": v.mean(axis=0).tolist(),
        "size": [float(np.linalg.norm(ex)), float(np.linalg.norm(ey)), float(np.linalg.norm(ez))],
        "x_axis": x.astype(np.float32).tolist(),
        "y_axis": y.astype(np.float32).tolist(),
    }


def edge_lengths(corners: Sequence[Sequence[float]]) -> np.ndarray:
    """(|v1-v0|, |v3-v0|, |v4-v0|): width, height and length of the box."""
    v = np.asarray(corners, dtype=np.float64)
    return np.linalg.norm(np.stack([v[1] - v[0], v[3] - v[0], v[4] - v[0]]), axis=1)


def axes_to_rotation_matrix(x_axis, y_axis) -> np.ndarray:
    """Orthonormal rotation [x | y | z] from two (possibly non-orthogonal) predicted axes (Gram-Schmidt)."""
    x = np.asarray(x_axis, dtype=np.float64)
    y = np.asarray(y_axis, dtype=np.float64)
    x = x / np.linalg.norm(x)
    y = y - np.dot(y, x) * x
    y_norm = np.linalg.norm(y)
    if y_norm < 1e-6:
        y_norm += 1e-4
    y = y / y_norm
    z = np.cross(x, y)
    z = z / np.linalg.norm(z)
    return np.stack([x, y, z], axis=1)


def local_axis_to_corners(center, size, x_axis, y_axis) -> np.ndarray:
    """(center, size, x_axis, y_axis) -> 8 corners in Omni3D order, shape (8, 3)."""
    R = axes_to_rotation_matrix(x_axis, y_axis)
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    canonical = np.array([
        [-hx, -hy, -hz], [+hx, -hy, -hz], [+hx, +hy, -hz], [-hx, +hy, -hz],
        [-hx, -hy, +hz], [+hx, -hy, +hz], [+hx, +hy, +hz], [-hx, +hy, +hz],
    ])
    return canonical @ R.T + np.asarray(center, dtype=np.float64)


def is_vec3(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(v, (int, float)) for v in value)
    )
