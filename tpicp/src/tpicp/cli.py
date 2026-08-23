"""Command-line entry point for standalone TP-ICP scale recovery."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from .core import TPICPConfig, solve_scale_translation_icp
from .io import load_best_ism_detection, load_camera, load_template_rotation, sample_rotated_model
from .pointcloud import prepare_observed_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--depth-path", type=Path, required=True)
    parser.add_argument("--camera-path", type=Path, required=True)
    parser.add_argument("--ism-result", type=Path, required=True)
    parser.add_argument("--template-poses", type=Path, required=True)
    parser.add_argument("--output-mesh", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--image-id", type=int)
    parser.add_argument("--mbsor-percentile", type=float, default=95.0)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--outlier-multiplier", type=float, default=3.0)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Match the production entry point: seed the legacy NumPy RNG once before
    # query preprocessing, then call Trimesh.sample for model surface sampling.
    np.random.seed(args.seed)
    camera, depth_scale = load_camera(args.camera_path, args.image_id)
    template_index, mask, detection_index = load_best_ism_detection(args.ism_result)
    rotation = load_template_rotation(args.template_poses, template_index)
    depth = np.asarray(Image.open(args.depth_path), dtype=np.float64)
    observed = prepare_observed_points(
        depth,
        mask,
        camera,
        depth_scale=depth_scale,
        mbsor_percentile=args.mbsor_percentile,
        sample_count=args.sample_count,
    )
    if len(observed) < 100:
        raise RuntimeError(f"too few observed points after preprocessing: {len(observed)}")

    mesh = trimesh.load(args.model_path, force="mesh")
    rotated_model = sample_rotated_model(
        mesh,
        rotation,
        count=args.sample_count,
    )
    config = TPICPConfig(
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        outlier_multiplier=args.outlier_multiplier,
    )
    result = solve_scale_translation_icp(observed, rotated_model, config)

    scaled_mesh = mesh.copy()
    scaled_mesh.apply_scale(result.scale)
    args.output_mesh.parent.mkdir(parents=True, exist_ok=True)
    scaled_mesh.export(args.output_mesh)

    payload = {
        **asdict(result),
        "translation": result.translation.tolist(),
        "template_index": template_index,
        "detection_index": detection_index,
        "num_points_observed": len(observed),
        "num_points_model": len(rotated_model),
        "mbsor_percentile": args.mbsor_percentile,
        "sample_count": args.sample_count,
        "sampling_seed": args.seed,
        "output_mesh": str(args.output_mesh),
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
