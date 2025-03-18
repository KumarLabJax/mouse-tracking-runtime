# Single Mouse Segmentation

Original Training Code: https://github.com/KumarLabJax/MouseTracking
Trained Models:
* Tracking Paper Model: https://zenodo.org/records/5806397
* High Quality Segmenter: (Not published)

## TSF Model

The segmentation model was exported using code that resides in the obj-api codebase. This code was largely based on tensorflow example code for optimizing and freezing a model.

# Single Mouse Pose

Original Training Code: https://github.com/KumarLabJax/deep-hrnet-mouse
Trained Models:
* Gait Paper Model: https://zenodo.org/records/6380163

## Pytorch Model

The pytorch model is the released model.

# Multi-Mouse Pose

Original Training Code: https://github.com/KumarLabJax/deep-hrnet-mouse
Trained Models:
* Top-down: In Progress
* Bottom-up: (Not published)

## Pytorch Model

The pytorch model is the model saved by the original hrnet code.

# Multi-Mouse Segmentation

Original Training Code: fork of https://github.com/google-research/deeplab2
Trained Models:
* Panoptic Deeplab: Not yet released

## TFS Model

deeplab2 provides `export_model.py`. This transforms the checkpoint into a tensorflow serving model.

```
python3 /deeplab2/deeplab2/export_model.py --checkpoint_path /deeplab2/trained_model/ckpt-125000 --experiment_option_path /deeplab2/trained_model/resnet50_os16.textproto --output_path tfs-models/multi-mouse-segmentation/panoptic-deeplab/
```

# Static Objects

## Arena Corners

Original Training Code: In Progress
Trained Models:
* Object Detection API (2022): Not yet released

### TFS Model

Export the model using the tf-obj-api exporter (in obj-api environment):
```
python /object_detection/models/research/object_detection/exporter_main_v2.py --input_type image_tensor --pipeline_config_path /object_detection/code/tf-obj-api/corner-detection/single-object-testing/pipeline.config --trained_checkpoint_dir /media/bgeuther/Storage/TempStorage/pose-validation/movenet/arena_corner/output_models/ --output_directory /media/bgeuther/Storage/TempStorage/trained-models/static-objects/obj-api-corner/
```
Note that this needs to be run in the folder with annotations if the config points to label_map.pbtxt locally.
`/media/bgeuther/Storage/TempStorage/pose-validation/movenet/arena_corner/` is the location of these annotations.

## Food Hopper

Original Training Code: In Progress
Trained Models:
* Object Detection API (2022): In Progress

### TFS Model

Export the model using the tf-obj-api exporter (in obj-api environment):
```
python /object_detection/models/research/object_detection/exporter_main_v2.py --input_type image_tensor --pipeline_config_path /object_detection/code/tf-obj-api/food-detection/segmentation/pipeline.config --trained_checkpoint_dir /media/bgeuther/Storage/TempStorage/pose-validation/movenet/food_hopper/output_models/ --output_directory /media/bgeuther/Storage/TempStorage/trained-models/static-objects/obj-api-food/
```
Note that this needs to be run in the folder with annotations if the config points to label_map.pbtxt locally.
`/media/bgeuther/Storage/TempStorage/pose-validation/movenet/food_hopper/` is the location of these annotations.

## Lixit

Original Training Code: In Progress
Trained Models:
* DeepLabCut: In Progress

# Dynamic Objects

## Fecal Boli

Original Training Code: https://github.com/KumarLabJax/deep-hrnet-mouse
Trained Models:
* fecal-boli (2020): Not yet published.

### Pytorch Model

The pytorch model is the model saved by the original hrnet code.
