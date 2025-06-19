# Multi-Mouse Pose Estimation Format (Version 7)

This document provides a comprehensive definition of the current Multi-Mouse Pose Estimation format.

## File Naming Convention

Each video has a corresponding HDF5 file with the same name as the corresponding video, replacing ".avi" with "_pose_est_v7.h5".

## Dataset Structure

### Core Pose Datasets

- `poseest/points`
  - Description: Keypoint pose information.
  - Shape: (#frames x #instance x #keypoints x 2)
    - #frames: the frame index of the video
    - #instance: the prediction instance (single animal) within the frame
    - #keypoints: the 12 keypoints described below in keypoint mapping section
    - 2: the pixel (y, x) position of the keypoint
  - Type: 16-bit unsigned integer
  - Padding: 0
  - Attributes:
    - `config`: details the name of the configuration file used during inference
    - `model`: details the saved model file used during inference
  - Notes:
    - Not all frames must have the same number of predicted instances. Padded instances are always the last indices in #instance dimension.
    - Not all instances must have all keypoints predicted. Padded keypoints will simply contain the padding value.

- `poseest/confidence`
  - Description: Confidence values (value of the peak on prediction heatmap) associated with keypoint pose data.
  - Shape: (#frames x #instances x #keypoints)
    - #frames: the frame index of the video
    - #instance: the prediction instance (single animal) within the frame
    - #keypoints: the 12 keypoints described below in keypoint mapping section
  - Type: 32-bit floating point
  - Padding: 0
  - Notes:
    - Index in this field corresponds to index in `poseest/points` field.
    - Anything higher than 0 indicates a valid point.

- `poseest/instance_count`
  - Description: Number of per-frame valid instances contained in `poseest/points` field.
  - Shape: (#frames)
    - #frames: the frame index of the video
  - Type: 8-bit unsigned integer
  - Padding: None
  - Notes:
    - This field provides a simplified count of instances in a frame that contain at least 1 non-zero keypoint confidence.

- `poseest/instance_embedding`
  - Description: Associative embedding value for keypoints.
  - Shape: (#frames x #instances x #keypoints)
    - #frames: the frame index of the video
    - #instance: the prediction instance (single animal) within the frame
    - #keypoints: the 12 keypoints described below in keypoint mapping section
  - Type: 32-bit floating point
  - Padding: 0
  - Notes:
    - This field is only typically used for bottom-up multi-mouse pose.

- `poseest/instance_track_id`
  - Description: Resolved tracklet key for an animal over time.
  - Shape: (#frames x #instances)
    - #frames: the frame index of the video
    - #instance: the prediction instance (single animal) within the frame
  - Type: 32-bit unsigned integer
  - Padding: 0
  - Notes:
    - Tracklets are continuous intervals of frames (no breaks/gaps).
    - Tracklets can start at index 0, meaning padded values must be masked with `poseest/instance_count` field.

### Identity Embedding Datasets

- `poseest/id_mask`
  - Description: Mask matrix for `poseest/instance_embed_id` field.
  - Shape: (#frames x #instances)
    - #frames: the frame index of the video
    - #instance: the prediction instance (single animal) within the frame
  - Type: bool
  - Padding: None
  - Notes:
    - Contains a 0 or 1 depending upon if the instance_embed_id is usable
    - Uses numpy masking convention, where 0 means "good data" and 1 means "data to ignore"
    - Instances marked as "good data" only include instances assigned a long-term ID
    - Instances marked as "data to ignore" include both invalid instances and instances not assigned a long-term ID

- `poseest/identity_embeds`
  - Description: Identity embeding value for each pose.
  - Shape: (#frames x #instances x embedded dimension)
    - #frames: the frame index of the video
    - #instance: the prediction instance (single animal) within the frame
    - embedded dimension: Size of bottleneck in identity embedding network
  - Type: 32-bit floating point
  - Padding: 0
  - Attributes:
    - `config`: details the name of the configuration file used during inference
    - `model`: details the saved model file used during inference

- `poseest/instance_embed_id`
  - Description: Resolved longterm identity for an animal over time.
  - Shape: (#frames x #instances)
    - #frames: the frame index of the video
    - #instance: the prediction instance (single animal) within the frame
  - Type: 32-bit unsigned integer
  - Padding: 0
  - Attributes:
    - `version`: version of the identity resolving code that generated these values
    - `tracklet_gen`: algorithm used for generating tracklets
    - `tracklet_stitch`: algorithm used for stitching together multiple tracklets
  - Notes:
    - number of clusters is indicated by shape of `poseest/instance_id_center`
    - Values of 0 are reserved for "non-valid" instances
    - Values of 1-number of clusters are long-term IDs
    - Values greater than number of clusters are valid instances/tracks that were not assigned an identity
    - Can be used with `posest/id_mask` to hide instances that were not assigned a long-term ID

- `poseest/instance_id_center`
  - Description: Center embedding location for each identity
  - Shape: (#clusters x embedded dimension)
    - #clusters: Number of identities assigned
    - embedded dimension: Size of bottleneck in identity embedding network
  - Type: 32-bit floating point
  - Padding: 0
  - Notes:
    - This field is used for linking together identities over multiple videos

### Static Object Datasets

- `static_objects/corners`
  - Description: Arena corner keypoints.
  - Shape: (4, 2)
    - 4: Keypoint index
    - 2: the pixel (x, y) position of the keypoint
  - Type: 16-bit unsigned integer
  - Padding: Field optional
  - Notes:
    - Corners do not guarantee any sorting

- `static_objects/lixit`
  - Description: Keypoint location of the tip of drinking water spouts.
  - Shape: (#lixit, 2)
    - #lixit: Lixit instance
    - 2: the pixel (y, x) position of the keypoint
  - Type: 32-bit floating point
  - Padding: Field optional

- `static_objects/food_hopper`
  - Description: Food hopper corner keypoints.
  - Shape: (4, 2)
  - Type: 16-bit unsigned integer
  - Padding: Field optional
  - Notes:
    - Corners are sorted to produce a valid polygon (e.g., clockwise ordering)

### Segmentation Datasets

- `poseest/seg_data`
  - Description: Segmentation predictions, stored in a compressed contour format
  - Size: (#frames, #instances, #max_contours, #max_contour_length, 2)
    - #frames: the frame index of the video
    - #instance: the prediction instance (single animal) within the frame
    - #max_contours: list of contours used to describe an instance
    - #max_contour_length: sequence of points in a contour
    - 2: the pixel (x, y) position of the contour point
  - Type: 32-bit signed integer
  - Padding: -1
  - Attributes:
    - `config`: details the name of the configuration file used during inference
    - `model`: details the saved model file used during inference
  - Notes:
    - This storage follows opencv convention for the last 3 dimensions.
    - Contour data is padded with -1s because these are invalid contour points.

- `poseest/seg_external_flag`
  - Description: Flag indicating if a segmentation is internal or external.
  - Shape: (#frames, #instances, #max_contours)
    - #frames: the frame index of the video
    - #instance: the prediction instance (single animal) within the frame
    - #max_contours: list of contours used to describe an instance
  - Type: bool
  - Padding: False
  - Notes:
    - Dataset stores whether or not a given contour in `poseest/seg_data` is external (True) or internal (False)

### Segmentation Linking Datasets
Fields are only available if segmentation linking was applied.

- `poseest/instance_seg_id`
  - Description: Segmentation data that has been matched with pose track data.
  - Size: (#frames, #instance)
    - #frames: the frame index of the video
    - #instance: the prediction instance (single animal) within the frame
  - Type: 32-bit unsigned integer
  - Padding: 0
  - Notes:
    - For each frame, values in this field and `poseest/instance_track_id` are matched for identity.
    - Segmentation that is not matched to a keypoint pose is not considered valid.

- `poseest/longterm_seg_id`
  - Description: Segmentation data that has been matched with pose longterm identity data.
  - Size: (#frames, #instance)
    - #frames: the frame index of the video
    - #instance: the prediction instance (single animal) within the frame
  - Type: 32-bit unsigned integer
  - Padding: 0
  - Notes:
    - For each frame, values in this field and `poseest/instance_embed_id` are matched for identity.
    - Segmentation that is not matched to a keypoint pose is not considered valid.

### Dynamic Objects Datasets
Objects that change over time.

Types of dynamic objects:

- Objects that change in location
- Objects that change in count

Characteristic of predictions:

- Predictions are not made every single frame. While the objects may be dynamic, they shouldn’t be as active as the mouse.

- `dynamic_objects/[object_name]/counts`
  - Description: Count of valid objects
  - Shape: (#frames)
- `dynamic_objects/[object_name]/points`
  - Description: Point data describing the object
  - Shape: (#frames, #objects, [#points to describe object], 2)
    - #frames: the frame index of `dynamic_objects/[object_name]/sample_indices`
    - #objects: instance of the object
    - #points to describe object: Optional, if more than 1 keypoint is necessary to describe the object
    - 2: x,y or y,x sorting may be different per-static object
- `dynamic_objects/[object_name]/sample_indices`
  - Description: Frame indices when each prediction was made
  - Shape: (#frames)

List of current dynamic objects:

- `fecal_boli`
  - Each object is described with 1 keypoint.
  - Object is stored in (y, x) location

## Attributes

The `poseest` group can have these attributes:

- `cm_per_pixel` (optional)
  - Defines how many centimeters a pixel of open field represents
  - Datatype is 32-bit floating point scalar

- `cm_per_pixel_source` (optional)
  - Defines how the "cm_per_pixel" value was set
  - Value will be one of "corner_detection", "manually_set" or "default_alignment"
  - Datatype is string scalar

## Keypoint Mapping

The 12 keypoint indexes have the following mapping to mouse body parts:

```
NOSE_INDEX = 0
LEFT_EAR_INDEX = 1
RIGHT_EAR_INDEX = 2
BASE_NECK_INDEX = 3
LEFT_FRONT_PAW_INDEX = 4
RIGHT_FRONT_PAW_INDEX = 5
CENTER_SPINE_INDEX = 6
LEFT_REAR_PAW_INDEX = 7
RIGHT_REAR_PAW_INDEX = 8
BASE_TAIL_INDEX = 9
MID_TAIL_INDEX = 10
TIP_TAIL_INDEX = 11
```

## Important Notes

1. Applications should not assume all files contain all datasets from all versions. The absence of fields does not imply the absence of objects in the arena.

2. Dynamic object predictions are not made every single frame. While the objects may be dynamic, they aren't as active as mice.

3. The way tracklets are generated ensures they are continuous intervals (no breaks/gaps). Some software depends on this (e.g., JABS as of v0.16.3).