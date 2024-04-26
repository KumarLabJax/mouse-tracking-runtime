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

# Models

## Single Mouse Segmentation

Original Training Code: https://github.com/KumarLabJax/MouseTracking
Trained Models:
* Tracking Paper Model: https://zenodo.org/records/5806397
* High Quality Segmenter: (Not published)


### TSF Model

The segmentation model was exported using code that resides in the obj-api codebase. This code was largely based on example code for optimizing and freezing a model.

### ORT Model

You can convert the tensorflow model into onnx format using the following command:

```
python -m tf2onnx.convert --saved-model tfs-models/single-mouse-segmentation/tracking-paper/ --output onnx-models/single-mouse-segmentation/tracking-paper.onnx --opset 18
```

There is a known issue related to the ConvTranspose op not being supported by ONNX-runtime on their CUDA provider, so it needs to run on the CPU. Why? It's apparently not popular enough of a layer. See https://github.com/microsoft/onnxruntime/issues/11312

As such, in the ONNX-runtime, this model will run between the GPU and CPU and as such will perform poorly.

## Single Mouse Pose

Original Training Code: https://github.com/KumarLabJax/deep-hrnet-mouse
Trained Models:
* Gait Paper Model: https://zenodo.org/records/6380163

### ORT Model

In the source repository, there is an `onnx` branch with example code under `tools/export-onnx.py`. Essentially, the model is loaded within the original environment followed by a call to `torch.onnx.export`.

## Multi-Mouse Pose

Original Training Code: https://github.com/KumarLabJax/deep-hrnet-mouse
Trained Models:
* Top-down: In Progress
* Bottom-up: (Not published)

### ORT Model

In the source repository, there is an `onnx` branch with example code under `tools/export-onnx.py`. Essentially, the model is loaded within the original environment followed by a call to `torch.onnx.export`.

## Multi-Mouse Segmentation

Original Training Code: fork of https://github.com/google-research/deeplab2
Trained Models:
* Panoptic Deeplab: Not yet released

### TFS Model

deeplab2 provides `export_model.py`. This transforms the checkpoint into a tensorflow serving model.

```
python3 /deeplab2/deeplab2/export_model.py --checkpoint_path /deeplab2/trained_model/ckpt-125000 --experiment_option_path /deeplab2/trained_model/resnet50_os16.textproto --output_path tfs-models/multi-mouse-segmentation/panoptic-deeplab/
```

## Static Objects

### Arena Corners

Original Training Code: In Progress
Trained Models:
* Object Detection API (2022): Not yet released

#### TFS Model

Export the model using the tf-obj-api exporter (in obj-api environment):
```
python /object_detection/models/research/object_detection/exporter_main_v2.py --input_type image_tensor --pipeline_config_path /object_detection/code/tf-obj-api/corner-detection/single-object-testing/pipeline.config --trained_checkpoint_dir /media/bgeuther/Storage/TempStorage/pose-validation/movenet/arena_corner/output_models/ --output_directory /media/bgeuther/Storage/TempStorage/trained-models/static-objects/obj-api-corner/
```
Note that this needs to be run in the folder with annotations if the config points to label_map.pbtxt locally.
`/media/bgeuther/Storage/TempStorage/pose-validation/movenet/arena_corner/` is the location of these annotations.

#### ORT Model

Convert the model over to onnx:
```
python -m tf2onnx.convert --saved-model /media/bgeuther/Storage/TempStorage/trained-models/static-objects/obj-api-corner/saved_model/ --output onnx-models/static-objects/obj-api-corners.onnx --opset 13
```

```
2024-04-01 09:55:41,353 - INFO - Model inputs: ['input_tensor']
2024-04-01 09:55:41,353 - INFO - Model outputs: ['detection_boxes', 'detection_boxes_strided', 'detection_classes', 'detection_keypoint_scores', 'detection_keypoints', 'detection_multiclass_scores', 'detection_scores', 'num_detections']
```

### Food Hopper

Original Training Code: In Progress
Trained Models:
* Object Detection API (2022): In Progress

#### TFS Model

Export the model using the tf-obj-api exporter (in obj-api environment):
```
python /object_detection/models/research/object_detection/exporter_main_v2.py --input_type image_tensor --pipeline_config_path /object_detection/code/tf-obj-api/food-detection/segmentation/pipeline.config --trained_checkpoint_dir /media/bgeuther/Storage/TempStorage/pose-validation/movenet/food_hopper/output_models/ --output_directory /media/bgeuther/Storage/TempStorage/trained-models/static-objects/obj-api-food/
```
Note that this needs to be run in the folder with annotations if the config points to label_map.pbtxt locally.
`/media/bgeuther/Storage/TempStorage/pose-validation/movenet/food_hopper/` is the location of these annotations.

#### ORT Model

Convert the model over to onnx:
```
python -m tf2onnx.convert --saved-model /media/bgeuther/Storage/TempStorage/trained-models/static-objects/obj-api-food/saved_model/ --output onnx-models/static-objects/obj-api-food.onnx --opset 13
```

```
2024-04-01 09:56:32,297 - INFO - Model inputs: ['input_tensor']
2024-04-01 09:56:32,297 - INFO - Model outputs: ['detection_boxes', 'detection_boxes_strided', 'detection_classes', 'detection_masks', 'detection_multiclass_scores', 'detection_scores', 'num_detections']
```

### Lixit

Original Training Code: In Progress
Trained Models:
* DeepLabCut: In Progress
