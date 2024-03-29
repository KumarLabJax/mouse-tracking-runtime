# ONNX-Runtime Pipelines

This is a collection of Kumar Lab pipelines converted over to ONNX-Runtime.
This is specifically NOT designed for training new models, but rather takes ONNX-exported models and runs inference on videos using them.

# Installation

Both virtual environments and singularity containers are supported.

## Virtual Environment

Python 3.10 venv tested

```
python3 -m venv onnx-runtime-venv
source onnx-runtime-venv/bin/activate
pip install -r requirements.txt
```

If you are running on a system with CUDA 12, you need to use a different onnx runtime (instructions from https://onnxruntime.ai/docs/install/)

```
pip install --upgrade onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

## Singularity Containers

See containers in vm folder.

# Models

## Single Mouse Segmentation

Original Training Code: https://github.com/KumarLabJax/MouseTracking
Trained Models:
* Tracking Paper Model: https://zenodo.org/records/5806397
* High Quality Segmenter: (Not published)

```
python -m tf2onnx.convert --saved-model /media/bgeuther/Storage/TempStorage/pose-validation/movenet/external/single_mouse_segmentation/ --output onnx-models/single-mouse-segmentation/tracking-paper.onnx --opset 17
```

## Single Mouse Pose

Original Training Code: https://github.com/KumarLabJax/deep-hrnet-mouse
Trained Models:
* Gait Paper Model: https://zenodo.org/records/6380163

## Multi-Mouse Pose

Original Training Code: https://github.com/KumarLabJax/deep-hrnet-mouse
Trained Models:
* Top-down: In Progress
* Bottom-up: (Not published)

## Multi-Mouse Segmentation

Original Training Code: In Progress
Trained Models:
* Panoptic Segmentation: In Progress

## Static Objects

### Arena Corners

Original Training Code: In Progress
Trained Models:
* Object Detection API (2022): In Progress

### Food Hopper

Original Training Code: In Progress
Trained Models:
* Object Detection API (2022): In Progress

### Lixit

Original Training Code: In Progress
Trained Models:
* DeepLabCut: In Progress

