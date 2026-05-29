# CAD-Free-6D-Pose-TP-ICP
Continuous Metric Scale Recovery for CAD-Free 6D Object Pose Estimation from a Single Reference Image


Code for our paper *Continuous Metric Scale Recovery for CAD-Free 6D Object Pose Estimation from a Single Reference Image*, submitted to The Visual Computer.

The full source code will be released upon acceptance of the paper.
![Qualitative results](Qualitative%20results%20on%20LINEMOD.png)
## About the method

We propose a CAD-free 6D object pose estimation pipeline that requires only a single reference RGB image. The core algorithm, Template-Point ICP (TP-ICP), recovers the metric scale of a single-view-generated 3D proxy mesh by jointly optimizing scale and translation under a fixed template-derived rotation.

The pipeline integrates three off-the-shelf components without retraining:

- SAM 3 — concept-driven segmentation
- SAM 3D — single-view 3D reconstruction
- SAM-6D — coarse-to-fine pose matching

TP-ICP serves as the bridge between the scale-ambiguous canonical-space proxy mesh and the metric-space query depth observation.
