"""I/O helpers for the standalone TP-ICP runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh


def load_camera(path: Path, image_id: int | None = None) -> tuple[np.ndarray, float]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if "cam_K" not in data:
        if image_id is None:
            raise ValueError("image_id is required for a BOP scene_camera.json")
        data = data[str(image_id)]
    camera = np.asarray(data["cam_K"], dtype=np.float64).reshape(3, 3)
    return camera, float(data.get("depth_scale", 1.0))


def load_best_ism_detection(path: Path) -> tuple[int, np.ndarray, int]:
    with np.load(path, allow_pickle=True) as data:
        required = {"segmentation", "best_template_idx"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"missing ISM fields: {sorted(missing)}")
        detection_index = (
            int(np.argmax(data["score"]))
            if "score" in data.files and len(data["score"])
            else 0
        )
        template_index = int(data["best_template_idx"][detection_index])
        mask = np.asarray(data["segmentation"][detection_index], dtype=bool)
    return template_index, mask, detection_index


def load_template_rotation(path: Path, template_index: int) -> np.ndarray:
    poses = np.load(path)
    if not 0 <= template_index < len(poses):
        raise IndexError(f"template index {template_index} outside [0, {len(poses)})")
    return np.asarray(poses[template_index, :3, :3], dtype=np.float64).T


def sample_rotated_model(
    mesh: trimesh.Trimesh,
    rotation: np.ndarray,
    count: int = 5000,
) -> np.ndarray:
    """Sample the model through the same Trimesh API as the production script."""

    points = np.asarray(mesh.sample(count), dtype=np.float64)
    return points @ np.asarray(rotation, dtype=np.float64).reshape(3, 3).T
