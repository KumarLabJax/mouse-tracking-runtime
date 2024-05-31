# Deployment Runtime Pipelines

This is a collection of Kumar Lab pipelines converted over to a flexible deployment runtime.
This is specifically NOT designed for training new models, but rather takes exported/frozen models and runs inference on videos using them.

This repository uses both ONNX-runtime (ORT) and Tensorflow Serving (TFS).

# Installation

Both virtual environments and singularity containers are supported.

## Virtual Environment

Python 3.10 venv tested

```
python3 -m venv runtime-venv
source runtime-venv/bin/activate
pip install -r requirements.txt
```

If you are running on a system with CUDA 12, you need to use a different onnxruntime-gpu wheel (instructions from https://onnxruntime.ai/docs/install/)

```
pip install --upgrade onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

## Singularity Containers

See containers in vm folder.

### Notes

This container is a bit hacky, because we match "system" installed cuda runtime for tensorflow and allow pytorch to use pypi libraries. Only the major version needs to match.

Tensorflow GPU runtime: https://www.tensorflow.org/install/source#gpu
Container runtime: (Check using `nvcc --version` inside container)
Pytorch runtime: https://pytorch.org/get-started/locally/

# Running Pipelines

There are currently 2 pipelines available. Single mouse open field assays and multi mouse longterm monitoring assays.

## Single Mouse Open Field Pipeline

These models were designed to operate with our open field assay recorded at JAX. We release this dataset at https://doi.org/10.7910/DVN/SAPNJG

For general usage, see [infer-single-pose-pipeline-v6.sh](infer-single-pose-pipeline.sh).

This pipeline consists of 4 model predictions:
1. Single Mouse Pose (gait paper)
2. Arena Corners (unpublished update from gait paper)
3. Single Mouse Segmentation (tracking paper)
4. Fecal Boli detection (unpublished)

TODO: Add link to Jake's star protocols.

## Multi Mouse Longterm Monitoring Pipeline

These models were designed to operate with our JABS data acquisition hardware. For specifications on data collection, see https://www.biorxiv.org/content/10.1101/2022.01.13.476229v2. For specifications on reproducing the hardware, see https://github.com/KumarLabJax/JABS-data-pipeline

For general usage, see [infer-multi-pose-pipeline-v6.sh](infer-multi-pose-pipeline.sh).

This pipeline consists of 6 model predictions and 1 algorithmic step:
1. Multi Mouse Segmentation (unpublished)
2. Multi Mouse Pose (unpublished)
3. Multi Mouse Identity Embedding (unpublished)
4. Tracklet Generation and Stitching (unpublished)
5. Arena Corners (unpublished update from gait paper)
6. Food Hopper (unpublished)
7. Lixit Spout (unpublished)

# Available Models

## Single Mouse Segmentation

Original Training Code: https://github.com/KumarLabJax/MouseTracking

Trained Models:
* Tracking Paper Model: https://zenodo.org/records/5806397
* High Quality Segmenter: Not yet published.

## Single Mouse Pose

Original Training Code: https://github.com/KumarLabJax/deep-hrnet-mouse

Trained Models:
* Gait Paper Model: https://zenodo.org/records/6380163

## Multi-Mouse Pose

Original Training Code: https://github.com/KumarLabJax/deep-hrnet-mouse

Trained Models:
* Top-down: Not yet published.
* Bottom-up: Not yet published.

## Multi-Mouse Segmentation

Original Training Code: Fork of https://github.com/google-research/deeplab2

Trained Models:
* Panoptic Deeplab: Not yet published.

## Static Objects

### Arena Corners

Original Training Code: Fork of https://github.com/tensorflow/models/tree/master/research/object_detection

Trained Models:
* Object Detection API (2022): Not yet published.

### Food Hopper

Original Training Code: Fork of https://github.com/tensorflow/models/tree/master/research/object_detection

Trained Models:
* Object Detection API (2022): Not yet published.

### Lixit

Original Training Code: https://github.com/DeepLabCut/DeepLabCut

Trained Models:
* DeepLabCut (2023): Not yet published.

## Dynamic Objects

### Fecal Boli

Original Training Code: https://github.com/KumarLabJax/deep-hrnet-mouse

Trained Models:
* fecal-boli (2020): Not yet published.
