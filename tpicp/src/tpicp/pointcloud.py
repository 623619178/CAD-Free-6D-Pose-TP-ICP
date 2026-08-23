"""Depth back-projection and paper-default point-cloud preprocessing."""

from __future__ import annotations

import numpy as np


def backproject_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    depth_scale: float = 1.0,
) -> np.ndarray:
    """Back-project masked depth pixels into metric camera coordinates."""

    depth_image = np.asarray(depth, dtype=np.float64) * float(depth_scale)
    object_mask = np.asarray(mask, dtype=bool)
    camera = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    if depth_image.shape != object_mask.shape:
        raise ValueError(
            f"depth and mask shapes differ: {depth_image.shape} vs {object_mask.shape}"
        )

    rows, columns = np.where(object_mask)
    z = depth_image[rows, columns]
    valid = np.isfinite(z) & (z > 0)
    rows, columns, z = rows[valid], columns[valid], z[valid]
    if len(z) == 0:
        return np.empty((0, 3), dtype=np.float64)

    fx, fy = camera[0, 0], camera[1, 1]
    cx, cy = camera[0, 2], camera[1, 2]
    if fx == 0 or fy == 0:
        raise ValueError("camera focal lengths must be non-zero")
    x = (columns - cx) * z / fx
    y = (rows - cy) * z / fy
    return np.column_stack((x, y, z))


def median_distance_filter(
    points: np.ndarray,
    percentile: float = 95.0,
    minimum_points: int = 100,
) -> np.ndarray:
    """Keep points below a distance percentile around the coordinate median."""

    result = np.asarray(points, dtype=np.float64)
    if len(result) <= minimum_points:
        return result.copy()
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    center = np.median(result, axis=0)
    distances = np.linalg.norm(result - center, axis=1)
    threshold = np.percentile(distances, percentile)
    return result[distances < threshold]


def deterministic_downsample(
    points: np.ndarray,
    count: int = 5000,
) -> np.ndarray:
    """Downsample with the legacy NumPy RNG used by the production script.

    The caller seeds ``np.random`` once before preprocessing, matching the
    paper's production TP-ICP entry point.
    """

    result = np.asarray(points, dtype=np.float64)
    if count <= 0:
        raise ValueError("count must be positive")
    if len(result) <= count:
        return result.copy()
    indices = np.random.choice(len(result), count, replace=False)
    return result[indices]


def prepare_observed_points(
    depth: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    depth_scale: float = 1.0,
    mbsor_percentile: float = 95.0,
    sample_count: int = 5000,
) -> np.ndarray:
    points = backproject_depth(depth, mask, camera_matrix, depth_scale)
    points = median_distance_filter(points, mbsor_percentile)
    return deterministic_downsample(points, sample_count)
