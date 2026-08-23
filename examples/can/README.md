# Bundled can proxy provenance

`proxy_mesh.ply` is the prepared, scale-ambiguous proxy used by the
single-query demo. It is not the BOP ground-truth CAD mesh.

## Reference-stage source

- Object: LINEMOD object 5, `can`.
- Reference RGB: LM test scene/object 5, frame 720.
- Foreground mask: the corresponding BOP visible ground-truth mask.
- Reconstruction: SAM 3D generated the object reconstruction, which was
  converted to a triangular mesh (`can_mesh.ply`).

## Benchmark coordinate preprocessing

For evaluation in the BOP object coordinate convention, the reconstructed mesh
was centered at its own vertex mean and aligned by a rotation-only procedure.
The rotation was estimated against normalized BOP CAD geometry; both source and
target were independently centered and normalized before rotation estimation,
so no CAD metric size was transferred. A single global pre-scale of `150` was
then applied to place all generated proxies in a common numerical range. TP-ICP
still estimates the per-query metric scale from the query depth observation.

This coordinate preprocessing is an offline benchmark convention. The BOP CAD
mesh is not consumed by `run_demo.sh`, first-pass ISM, TP-ICP, fine ISM, or PEM.

## Query used by the demo

- Dataset frame: LMO scene 2, image 3.
- RGB, depth, and intrinsics: `SAM-6D/Data/Example/`.
- Target object: object 5, `can`.

Thus, the reference and query are distinct observations of the same object.

## Integrity

SHA-256 of the bundled proxy:

```text
455ca9ca6328fd23c0fc4baf49b1ff5f385414f118ed4a22d64bede1f11da388
```
