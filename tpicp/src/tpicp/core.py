"""Core fixed-rotation TP-ICP solver.

The correspondence direction matches the production implementation: for each
observed target point, find the nearest currently transformed model point.
Rotation is applied before this function and remains fixed throughout TP-ICP.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.spatial import KDTree


@dataclass(frozen=True)
class TPICPConfig:
    """Paper-default TP-ICP optimization settings."""

    max_iterations: int = 80
    tolerance: float = 1e-4
    outlier_multiplier: float = 3.0
    minimum_correspondences: int = 10

    def validate(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.outlier_multiplier <= 0:
            raise ValueError("outlier_multiplier must be positive")
        if self.minimum_correspondences < 3:
            raise ValueError("minimum_correspondences must be at least 3")


@dataclass(frozen=True)
class TPICPResult:
    scale: float
    translation: np.ndarray
    num_iterations: int
    num_correspondences_final: int
    converged: bool
    stop_reason: str
    runtime_sec: float


def _as_points(name: str, points: np.ndarray) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {result.shape}")
    if len(result) == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite points")
    return result


def initial_scale_translation(
    observed_points: np.ndarray,
    rotated_model_points: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Initialize scale from median radii and align point centroids."""

    observed = _as_points("observed_points", observed_points)
    model = _as_points("rotated_model_points", rotated_model_points)
    observed_center = observed.mean(axis=0)
    model_center = model.mean(axis=0)
    observed_radius = np.median(np.linalg.norm(observed - observed_center, axis=1))
    model_radius = np.median(np.linalg.norm(model - model_center, axis=1))
    scale = float(observed_radius / model_radius) if model_radius > 0 else 1.0
    translation = observed_center - scale * model_center
    return scale, translation


def solve_scale_translation_icp(
    observed_points: np.ndarray,
    rotated_model_points: np.ndarray,
    config: TPICPConfig | None = None,
    initial_scale: float | None = None,
    initial_translation: np.ndarray | None = None,
) -> TPICPResult:
    """Estimate isotropic scale and translation with fixed model rotation."""

    settings = config or TPICPConfig()
    settings.validate()
    observed = _as_points("observed_points", observed_points)
    model = _as_points("rotated_model_points", rotated_model_points)
    if len(observed) < settings.minimum_correspondences:
        raise ValueError("too few observed points")
    if len(model) < settings.minimum_correspondences:
        raise ValueError("too few model points")

    default_scale, default_translation = initial_scale_translation(observed, model)
    scale = default_scale if initial_scale is None else float(initial_scale)
    translation = (
        default_translation
        if initial_translation is None
        else np.asarray(initial_translation, dtype=np.float64).reshape(3)
    )
    if not np.isfinite(scale) or not np.all(np.isfinite(translation)):
        raise ValueError("initial scale and translation must be finite")

    started = time.perf_counter()
    final_correspondences = 0
    iterations = 0
    converged = False
    stop_reason = "max_iterations"

    for iteration in range(settings.max_iterations):
        transformed_model = scale * model + translation
        distances, indices = KDTree(transformed_model).query(observed)
        median_residual = float(np.median(distances))
        threshold = settings.outlier_multiplier * median_residual
        if threshold == 0:
            threshold = 1e-6
        inliers = distances < threshold
        final_correspondences = int(inliers.sum())
        if final_correspondences < settings.minimum_correspondences:
            stop_reason = "too_few_correspondences"
            break

        target = observed[inliers]
        source = model[indices[inliers]]
        target_center = target.mean(axis=0)
        source_center = source.mean(axis=0)
        target_centered = target - target_center
        source_centered = source - source_center
        denominator = float(
            np.sum(np.sum(source_centered * source_centered, axis=1))
        )
        if not np.isfinite(denominator) or denominator <= 0:
            stop_reason = "degenerate_source"
            break

        numerator = float(
            np.sum(np.sum(target_centered * source_centered, axis=1))
        )
        new_scale = numerator / denominator
        new_translation = target_center - new_scale * source_center
        if not np.isfinite(new_scale) or not np.all(np.isfinite(new_translation)):
            stop_reason = "non_finite_update"
            break

        iterations = iteration + 1
        scale_change = abs(new_scale - scale)
        scale = new_scale
        translation = new_translation
        if scale_change < settings.tolerance:
            converged = True
            stop_reason = "scale_tolerance"
            break

    return TPICPResult(
        scale=scale,
        translation=translation,
        num_iterations=iterations,
        num_correspondences_final=final_correspondences,
        converged=converged,
        stop_reason=stop_reason,
        runtime_sec=time.perf_counter() - started,
    )
