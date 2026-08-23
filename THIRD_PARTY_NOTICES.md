# Third-Party Notices

This repository combines project-owned TP-ICP code with third-party visual
computing components. Third-party code must not be described as an original
contribution of this paper.

## SAM-6D

- Upstream: https://github.com/JiehongLin/SAM-6D
- Audited source revision: `1c2543b3b6faa1f1d81b3c7291f8b371d71e50c2`
- Role: instance segmentation, template matching, template rendering, and pose
  estimation.

The bundled SAM-6D source preserves upstream attribution and license files.
The custom inference entry points are adapted to save `best_template_idx` for
TP-ICP initialization and to support the single-query integration. SAM-6D is
not claimed as project-owned code.

## Other components

- SAM 3: https://github.com/facebookresearch/sam3
- SAM 3D Objects: https://github.com/facebookresearch/sam-3d-objects
- BOP Toolkit: https://github.com/thodan/bop_toolkit
- BlenderProc: https://github.com/DLR-RM/BlenderProc
- Segment Anything: https://github.com/facebookresearch/segment-anything
- DINOv2: https://github.com/facebookresearch/dinov2

Use each component and checkpoint under its corresponding upstream terms.

## Checkpoints and datasets

Pretrained checkpoints and full LM/LMO datasets are excluded from Git. The
example RGB-D input is inherited from the SAM-6D example package. The bundled
proxy mesh is a generated paper artifact provided only for reproducing the
single-query demo.

## Release check

Before public redistribution, the authors must manually confirm the applicable
license terms for every bundled SAM-6D subdirectory and add a project-level
license for the original TP-ICP code.
