# Multi-Mouse Pose Estimation Format (Version 7)

This document provides a comprehensive definition of the current Multi-Mouse Pose Estimation format.

## File Naming Convention

Each video has a corresponding HDF5 file with the same name as the corresponding video, replacing ".avi" with "_pose_est_v7.h5".

## Dataset Structure

### Core Pose Datasets

- `poseest/points`
  - Dataset with size (#frames x maximum # instances x #keypoints x 2)
  - #keypoints is 12 following the mouse body part indexing scheme
  - The last dimension of size 2 holds the pixel (y, x) position
  - Datatype is 16-bit unsigned integer

- `poseest/confidence`
  - Dataset with size (#frames x maximum # instances x #keypoints)
  - Assigns a confidence value to each of the 12 points
  - Values of 0 indicate a missing point; anything higher than 0 indicates a valid point
  - Datatype is 32-bit floating point

- `poseest/instance_count`
  - Dataset with size (#frames)
  - Gives the instance count for every frame
  - Datatype is 8-bit unsigned integer

- `poseest/instance_embedding`
  - Dataset with size (#frames x maximum # instances x #keypoints)
  - Contains the instance embedding for the respective instance at the respective frame and point
  - Datatype is 32-bit floating point

- `poseest/instance_track_id`
  - Dataset with size (#frames x maximum # instances)
  - Contains the instance_track_id for each instance index on a per-frame basis
  - Tracklets are continuous blocks (no breaks/gaps)

### Identity Embedding Datasets

- `poseest/id_mask`
  - Dataset with size (#frames x maximum # instances)
  - Contains a 0 or 1 depending upon if the instance_embed_id is usable
  - Uses numpy masking convention, where 0 means "good data" and 1 means "data to ignore"
  - Instances marked as "good data" only include instances assigned a long-term ID
  - Instances marked as "data to ignore" include both invalid instances and instances not assigned a long-term ID

- `poseest/identity_embeds`
  - Dataset with size (#frames x maximum # instances x embedded dimension)
  - Contains the embedded identity location for each pose

- `poseest/instance_embed_id`
  - Dataset with size (#frames x maximum # instances)
  - A corrected "poseest/instance_track_id"
  - Can be used with "posest/id_mask" to hide instances that were not assigned a long-term ID
  - Values of 0 are reserved for "non-valid" instances
  - Values of 1-# clusters are long-term IDs
  - Values greater than # clusters are valid instances/tracks that were not assigned an identity
  - Optional attributes store information about the method used for generation:
    - `version` = current version of the code run
    - `tracklet_gen` = algorithm used for generating the tracklets
    - `tracklet_stitch` = algorithm used for stitching together multiple tracklets

- `poseest/instance_id_center`
  - Dataset with size (# clusters x embedded dimension)
  - Contains the embedded locations of the clusters
  - Typically used for linking together identities over multiple videos

### Static Object Datasets

- `static_objects/corners`
  - Dataset with shape (4, 2), dtype=16-bit unsigned integer
  - Contains x,y coordinates of the 4 arena corners
  - Corners do not guarantee any sorting

- `static_objects/lixit`
  - Dataset with shape (n, 2), dtype=32-bit float
  - Contains y,x coordinates for each of n lixit water spouts

- `static_objects/food_hopper`
  - Dataset with shape (4,2), dtype=16-bit unsigned integer
  - Contains y,x coordinates for the 4 detected corners of the food hopper
  - Corners are sorted to produce a valid polygon (e.g., clockwise ordering)

### Segmentation Datasets

- `poseest/seg_data`
  - Dataset with size (#frames, #max_animals, #max_contours, #max_contour_length, 2)
  - Dataset type is int32
  - Dataset is padded with -1s to complete the tensor, since not all contours may exist or be the same size
  - For dimension 3, each animal can be described as multiple contours (external or internal)
  - Attribute `config` details the name of the configuration file used during inference
  - Attribute `model` details the saved model file used during inference

- `poseest/seg_external_flag`
  - Dataset with size (#frames, #max_animals, #max_contours)
  - Dataset type is bool
  - Dataset stores whether or not a given contour is external (True) or internal (False)

### Segmentation Linking Datasets
Fields are only available if segmentation linking was applied.

- `poseest/instance_seg_id`
  - Dataset with size (#frames, #max_animals)
  - Represents tracklets generated for segmentations
  - Values the same across both this field and "instance_track_id" identify animals with BOTH a pose prediction and a segmentation prediction

- `poseest/longterm_seg_id`
  - Dataset with size (#frames, #max_animals)
  - Represents corrected tracklets
  - Values the same across both this field and "instance_embed_id" identify animals with BOTH a pose prediction and a segmentation prediction
  - Relates to `poseest/instance_id_center`

### Dynamic Objects Datasets
Objects that change over time.

Types of dynamic objects:

- Objects that change in location
- Objects that change in count

Characteristic of predictions:

- Predictions are not made every single frame. While the objects may be dynamic, they shouldn’t be as active as the mouse.

- `dynamic_objects/[object_name]/counts`
  - Count of valid objects

- `dynamic_objects/[object_name]/points`
  - Point data describing the object
  - x,y or y,x sorting may be different per-static object

- `dynamic_objects/[object_name]/sample_indices`
  - Frame indices when each prediction was made

## Attributes

The "poseest" group can have these attributes:

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

## Dynamic Object Keypoint Sorting

- `fecal_boli`: Sorting is `y, x`, since this network was created using HRNet.

## Important Notes

1. Applications should not assume all files contain all datasets from all versions. The absence of fields does not imply the absence of objects in the arena.

2. Dynamic object predictions are not made every single frame. While the objects may be dynamic, they aren't as active as mice.

3. The way tracklets are generated ensures they are continuous blocks (no breaks/gaps). Some software depends on this (e.g., JABS as of v0.16.3).