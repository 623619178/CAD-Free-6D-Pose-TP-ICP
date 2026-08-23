# CAD-Free-6D-Pose-TP-ICP

Code for the paper **Continuous Metric Scale Recovery for CAD-Free 6D Object
Pose Estimation from a Single Reference Image**.

![Qualitative results](Qualitative%20results%20on%20LINEMOD.png)

## About

This repository provides the implementation of TP-ICP for continuous metric
scale recovery and its integration with SAM-6D for 6D object pose estimation.
TP-ICP estimates scale and translation for a CAD-free 3D proxy before final
pose estimation.

## Setup

Create the SAM-6D environment and download the official checkpoints:

```bash
cd SAM-6D
sh prepare.sh
cd ..
conda activate sam6d
```

## Inference

```bash
bash run_demo.sh
```

The main outputs are:

```text
outputs/can_demo/final_result.json
outputs/can_demo/05_pem/sam6d_results/vis_pem.png
```

## Acknowledgments

This work builds on the following open-source projects:

- [SAM 3](https://github.com/facebookresearch/sam3)
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)
- [SAM-6D](https://github.com/JiehongLin/SAM-6D)
- [BOP Toolkit](https://github.com/thodan/bop_toolkit)

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution details.
