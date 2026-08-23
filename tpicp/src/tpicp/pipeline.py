"""Run one complete SAM-6D query with TP-ICP metric scale recovery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence


class PipelineError(RuntimeError):
    """Raised when a required pipeline stage does not produce its output."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run initial ISM, TP-ICP, scaled-template rendering, fine ISM, "
            "and PEM for one RGB-D query."
        )
    )
    parser.add_argument("--sam6d-root", type=Path, required=True)
    parser.add_argument("--proxy-mesh", type=Path, required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--initial-templates-dir",
        type=Path,
        help=(
            "Prepared unscaled template directory. It may be either the "
            "directory containing rgb_*.png or its parent containing templates/. "
            "When omitted, the templates are rendered from --proxy-mesh."
        ),
    )
    parser.add_argument("--template-poses", type=Path)
    parser.add_argument("--category-id", type=int)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--segmentor-model", default="sam")
    parser.add_argument("--det-score-thresh", type=float, default=0.32)
    parser.add_argument("--mbsor-percentile", type=float, default=95.0)
    parser.add_argument("--sample-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--outlier-multiplier", type=float, default=3.0)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--blenderproc", default="blenderproc")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory. The default is to fail safely.",
    )
    return parser


def _require_file(path: Path, label: str) -> Path:
    result = path.expanduser().resolve()
    if not result.is_file():
        raise PipelineError(f"missing {label}: {result}")
    return result


def _require_directory(path: Path, label: str) -> Path:
    result = path.expanduser().resolve()
    if not result.is_dir():
        raise PipelineError(f"missing {label}: {result}")
    return result


def _template_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("rgb_*.png"))


def _validate_templates(directory: Path, expected_count: int = 42) -> None:
    rgb_files = _template_files(directory)
    if len(rgb_files) != expected_count:
        raise PipelineError(
            f"expected {expected_count} rgb templates in {directory}, found {len(rgb_files)}"
        )
    for rgb_path in rgb_files:
        index = rgb_path.stem.removeprefix("rgb_")
        for prefix, suffix in (("mask_", ".png"), ("xyz_", ".npy")):
            companion = directory / f"{prefix}{index}{suffix}"
            if not companion.is_file():
                raise PipelineError(f"template companion is missing: {companion}")


def _resolve_templates(path: Path) -> Path:
    candidate = _require_directory(path, "initial templates directory")
    if _template_files(candidate):
        _validate_templates(candidate)
        return candidate
    nested = candidate / "templates"
    if nested.is_dir() and _template_files(nested):
        _validate_templates(nested)
        return nested
    raise PipelineError(
        f"no rgb_*.png templates found in {candidate} or {nested}"
    )


def _copy_templates(source: Path, destination: Path) -> None:
    if destination.exists():
        raise PipelineError(f"template destination already exists: {destination}")
    shutil.copytree(source, destination, symlinks=True)


def _run_command(
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str],
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n")
        log.write(f"CWD: {cwd}\n\n")
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    if return_code != 0:
        raise PipelineError(
            f"command failed with exit code {return_code}; see {log_path}"
        )
    return elapsed


def _load_json(path: Path, label: str) -> Any:
    _require_file(path, label)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def select_best_detection(
    detections: Any,
    category_id: int | None = None,
) -> dict[str, Any]:
    if not isinstance(detections, list) or not detections:
        raise PipelineError("PEM returned no detections")
    candidates = detections
    if category_id is not None:
        candidates = [
            detection
            for detection in detections
            if int(detection.get("category_id", detection.get("obj_id", -1)))
            == category_id
        ]
        if not candidates:
            raise PipelineError(f"PEM returned no detection for category {category_id}")
    best = max(candidates, key=lambda detection: float(detection.get("score", 0.0)))
    for field in ("R", "t", "score"):
        if field not in best:
            raise PipelineError(f"best PEM detection is missing {field}")
    return best


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sam6d_root = _require_directory(args.sam6d_root, "SAM-6D root")
    proxy_mesh = _require_file(args.proxy_mesh, "proxy mesh")
    rgb = _require_file(args.rgb, "query RGB image")
    depth = _require_file(args.depth, "query depth image")
    camera = _require_file(args.camera, "camera JSON")

    ism_root = _require_directory(
        sam6d_root / "Instance_Segmentation_Model", "SAM-6D ISM directory"
    )
    pem_root = _require_directory(
        sam6d_root / "Pose_Estimation_Model", "SAM-6D PEM directory"
    )
    render_root = _require_directory(sam6d_root / "Render", "SAM-6D Render directory")
    ism_script = _require_file(ism_root / "run_inference_custom.py", "ISM runner")
    pem_script = _require_file(pem_root / "run_inference_custom.py", "PEM runner")
    render_script = _require_file(
        render_root / "render_custom_templates.py", "template renderer"
    )
    template_poses = args.template_poses or (
        ism_root / "utils/poses/predefined_poses/cam_poses_level0.npy"
    )
    template_poses = _require_file(template_poses, "template camera poses")

    output_root = args.output_dir.expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise PipelineError(
                f"output directory already exists: {output_root}; use --overwrite explicitly"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    result_path = output_root / "final_result.json"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    timings: dict[str, float] = {}
    stage = "initialization"
    total_started = time.perf_counter()

    try:
        first_ism_dir = output_root / "01_first_ism"
        first_ism_dir.mkdir()
        if args.initial_templates_dir:
            stage = "copy_initial_templates"
            initial_templates = _resolve_templates(args.initial_templates_dir)
        else:
            stage = "render_initial_templates"
            initial_render_dir = output_root / "00_initial_render"
            timings[stage] = _run_command(
                [
                    args.blenderproc,
                    "run",
                    str(render_script),
                    "--output_dir",
                    str(initial_render_dir),
                    "--cad_path",
                    str(proxy_mesh),
                ],
                render_root,
                output_root / "logs/00_initial_render.log",
                env,
            )
            initial_templates = _resolve_templates(initial_render_dir)
        _copy_templates(initial_templates, first_ism_dir / "templates")

        stage = "first_pass_ism"
        timings[stage] = _run_command(
            [
                sys.executable,
                str(ism_script),
                "--segmentor_model",
                args.segmentor_model,
                "--output_dir",
                str(first_ism_dir),
                "--cad_path",
                str(proxy_mesh),
                "--rgb_path",
                str(rgb),
                "--depth_path",
                str(depth),
                "--cam_path",
                str(camera),
            ],
            ism_root,
            output_root / "logs/01_first_ism.log",
            env,
        )
        first_ism_npz = _require_file(
            first_ism_dir / "sam6d_results/detection_ism.npz",
            "first-pass ISM NPZ",
        )

        stage = "tpicp"
        tpicp_dir = output_root / "02_tpicp"
        tpicp_dir.mkdir()
        scaled_mesh = tpicp_dir / "scaled_proxy_mesh.ply"
        tpicp_json = tpicp_dir / "tpicp_result.json"
        timings[stage] = _run_command(
            [
                sys.executable,
                "-m",
                "tpicp.cli",
                "--model-path",
                str(proxy_mesh),
                "--depth-path",
                str(depth),
                "--camera-path",
                str(camera),
                "--ism-result",
                str(first_ism_npz),
                "--template-poses",
                str(template_poses),
                "--output-mesh",
                str(scaled_mesh),
                "--output-json",
                str(tpicp_json),
                "--mbsor-percentile",
                str(args.mbsor_percentile),
                "--sample-count",
                str(args.sample_count),
                "--seed",
                str(args.seed),
                "--outlier-multiplier",
                str(args.outlier_multiplier),
                "--max-iterations",
                str(args.max_iterations),
                "--tolerance",
                str(args.tolerance),
            ],
            output_root,
            output_root / "logs/02_tpicp.log",
            env,
        )
        _require_file(scaled_mesh, "scaled proxy mesh")
        tpicp_result = _load_json(tpicp_json, "TP-ICP result JSON")

        stage = "render_scaled_templates"
        scaled_render_dir = output_root / "03_scaled_render"
        timings[stage] = _run_command(
            [
                args.blenderproc,
                "run",
                str(render_script),
                "--output_dir",
                str(scaled_render_dir),
                "--cad_path",
                str(scaled_mesh),
            ],
            render_root,
            output_root / "logs/03_scaled_render.log",
            env,
        )
        scaled_templates = _resolve_templates(scaled_render_dir)

        stage = "fine_ism"
        fine_ism_dir = output_root / "04_fine_ism"
        fine_ism_dir.mkdir()
        _copy_templates(scaled_templates, fine_ism_dir / "templates")
        timings[stage] = _run_command(
            [
                sys.executable,
                str(ism_script),
                "--segmentor_model",
                args.segmentor_model,
                "--output_dir",
                str(fine_ism_dir),
                "--cad_path",
                str(scaled_mesh),
                "--rgb_path",
                str(rgb),
                "--depth_path",
                str(depth),
                "--cam_path",
                str(camera),
            ],
            ism_root,
            output_root / "logs/04_fine_ism.log",
            env,
        )
        fine_ism_json = _require_file(
            fine_ism_dir / "sam6d_results/detection_ism.json",
            "fine ISM detection JSON",
        )

        stage = "pem"
        pem_dir = output_root / "05_pem"
        pem_dir.mkdir()
        _copy_templates(scaled_templates, pem_dir / "templates")
        timings[stage] = _run_command(
            [
                sys.executable,
                str(pem_script),
                "--output_dir",
                str(pem_dir),
                "--cad_path",
                str(scaled_mesh),
                "--rgb_path",
                str(rgb),
                "--depth_path",
                str(depth),
                "--cam_path",
                str(camera),
                "--seg_path",
                str(fine_ism_json),
                "--det_score_thresh",
                str(args.det_score_thresh),
            ],
            pem_root,
            output_root / "logs/05_pem.log",
            env,
        )
        pem_json = pem_dir / "sam6d_results/detection_pem.json"
        detections = _load_json(pem_json, "PEM detection JSON")
        best = select_best_detection(detections, args.category_id)

        timings["total"] = time.perf_counter() - total_started
        payload = {
            "status": "ok",
            "scale": float(tpicp_result["scale"]),
            "tpicp_translation": tpicp_result["translation"],
            "R": best["R"],
            "t": best["t"],
            "score": float(best["score"]),
            "category_id": best.get("category_id", best.get("obj_id")),
            "timing_sec": timings,
            "inputs": {
                "proxy_mesh": str(proxy_mesh),
                "rgb": str(rgb),
                "depth": str(depth),
                "camera": str(camera),
            },
            "outputs": {
                "scaled_mesh": str(scaled_mesh),
                "first_ism": str(first_ism_npz),
                "fine_ism": str(fine_ism_json),
                "pem": str(pem_json),
            },
        }
        _write_result(result_path, payload)
        print(json.dumps(payload, indent=2))
        return 0
    except Exception as error:
        payload = {
            "status": "error",
            "failed_stage": stage,
            "message": str(error),
            "timing_sec": {
                **timings,
                "total": time.perf_counter() - total_started,
            },
        }
        _write_result(result_path, payload)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
