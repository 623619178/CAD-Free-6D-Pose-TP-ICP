# Third-Party Components

This directory contains only the project-owned TP-ICP implementation and the
single-query orchestration code. It does not redistribute the projects or
assets listed below.

## Runtime dependencies

- SAM-6D: https://github.com/JiehongLin/SAM-6D
- SAM 3: https://github.com/facebookresearch/sam3
- SAM 3D Objects: https://github.com/facebookresearch/sam-3d-objects
- BlenderProc: https://github.com/DLR-RM/BlenderProc

Use pinned revisions in the final public release and follow the license and
checkpoint terms of each upstream project. The local
`run_inference_custom.py` entry points are adaptations of SAM-6D integration
code. They must be released either in an attributed SAM-6D fork or as an
explicit patch against a pinned upstream revision; they must not be described
as original TP-ICP code.

## Data and checkpoints

LINEMOD/LMO images, depth maps, annotations, pretrained checkpoints, generated
proxy meshes, and template caches are not included. Users must obtain datasets
and checkpoints from their official sources. A generated proxy mesh may be
provided separately as a paper release asset after its redistribution status
has been confirmed.

## Project license

The authors must select and add a project-level license before publishing this
package. This notice does not grant rights to any third-party component.
