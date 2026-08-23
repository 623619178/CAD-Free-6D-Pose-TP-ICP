# TP-ICP Single-Query Pipeline

This package provides the project-owned TP-ICP implementation and a single
command that runs the complete query-stage pipeline used in the paper:

> Continuous Metric Scale Recovery for CAD-Free 6D Object Pose Estimation
> from a Single Reference Image

Given a prepared, scale-ambiguous proxy mesh and one RGB-D query, the command
runs:

1. initial 42-view template rendering (or reuses prepared templates),
2. first-pass SAM-6D ISM for the query mask and template-derived rotation,
3. TP-ICP metric scale recovery,
4. 42-view rendering of the scaled proxy,
5. fine SAM-6D ISM, and
6. SAM-6D PEM for the final 6D pose.

The final JSON contains the recovered scale, final rotation, translation,
confidence score, stage runtimes, and paths to all intermediate artifacts.

## Reproducibility boundary

This is a single-query inference release, not a dataset benchmark runner. It
starts from a prepared proxy mesh. Reference-image segmentation and SAM 3D
proxy reconstruction are external preprocessing steps. LM/LMO datasets,
pretrained checkpoints, and rendered-template caches are not redistributed by
this package. In the paper release repository, the compatible SAM-6D source and
one generated example proxy are bundled separately from this Python package.

SAM 3, SAM 3D, and SAM-6D are third-party components used without retraining.
Install them from their official repositories and comply with their licenses.
The `--sam6d-root` directory must contain:

```text
Instance_Segmentation_Model/run_inference_custom.py
Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy
Pose_Estimation_Model/run_inference_custom.py
Render/render_custom_templates.py
```

## Paper configuration

- fixed template-derived initial rotation,
- 5,000 observed/model points,
- 95th-percentile median-distance depth filtering,
- observed-to-transformed-model nearest-neighbor correspondences,
- correspondence threshold: 3 times the median residual,
- maximum iterations: 80,
- scale convergence tolerance: `1e-4`,
- 42 rendered templates,
- PEM detection threshold: `0.32`, and
- random seed: 2.

## Installation

Activate the environment in which SAM-6D and BlenderProc already run, then:

```bash
cd tpicp
python -m pip install -e .
```

Verify the project-owned TP-ICP implementation:

```bash
python -m unittest discover -s tests -v
```

## Required inputs

```text
proxy_mesh.ply           # scale-ambiguous proxy in the expected coordinates
query_rgb.png
query_depth.png          # depth units interpreted through depth_scale
camera.json              # cam_K and depth_scale
```

The camera file must be a JSON object such as:

```json
{
  "cam_K": [572.4, 0.0, 325.3, 0.0, 573.6, 242.0, 0.0, 0.0, 1.0],
  "depth_scale": 1.0
}
```

Prepared initial templates can be supplied with
`--initial-templates-dir`. Otherwise they are rendered once from the proxy.

## Run one complete query

```bash
tpicp-single-query \
  --sam6d-root /path/to/SAM-6D \
  --proxy-mesh /path/to/proxy_mesh.ply \
  --rgb /path/to/query_rgb.png \
  --depth /path/to/query_depth.png \
  --camera /path/to/camera.json \
  --output-dir outputs/example_query \
  --cuda-device 0
```

To reuse prepared unscaled templates:

```bash
tpicp-single-query \
  --sam6d-root /path/to/SAM-6D \
  --proxy-mesh /path/to/proxy_mesh.ply \
  --initial-templates-dir /path/to/unscaled/templates \
  --rgb /path/to/query_rgb.png \
  --depth /path/to/query_depth.png \
  --camera /path/to/camera.json \
  --output-dir outputs/example_query \
  --cuda-device 0
```

The runner refuses to replace an existing output directory unless
`--overwrite` is explicitly provided.

## Outputs

```text
outputs/example_query/
  00_initial_render/       # omitted when prepared templates are supplied
  01_first_ism/
  02_tpicp/
    scaled_proxy_mesh.ply
    tpicp_result.json
  03_scaled_render/
  04_fine_ism/
  05_pem/
  logs/
  final_result.json
```

The final result contains at least `status`, `scale`, `R`, `t`, `score`,
`timing_sec`, and paths to the retained intermediate results.

## TP-ICP-only command

For diagnosing the proposed scale solver without running downstream SAM-6D:

```bash
tpicp-scale \
  --model-path /path/to/proxy_mesh.ply \
  --depth-path /path/to/depth.png \
  --camera-path /path/to/camera.json \
  --ism-result /path/to/detection_ism.npz \
  --template-poses /path/to/cam_poses_level0.npy \
  --output-mesh /path/to/scaled_mesh.ply \
  --output-json /path/to/tpicp_result.json
```

## Code ownership

The `src/tpicp` scale-recovery implementation and query orchestration are the
project-owned release. SAM 3, SAM 3D, SAM-6D, BlenderProc, and BOP datasets are
external projects/assets and must not be presented as original code. See
`THIRD_PARTY_NOTICES.md` before redistribution.
