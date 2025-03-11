# Single Mouse

This pipeline currently includes the following deep learning models:
* Single Mouse Keypoints
* Single Mouse Segmentation
* Arena Corners
* Fecal Boli

Features generated from this pipeline include:
* pose_v2 related features
 * Gait features
 * Morphometric features
* pose_v6 related features (requires arena corners)
 * JABS-related features
 * Fecal boli count prediction for every minute

This pipeline also includes:
* Time-cropping of data (aligning to when the mouse is placed into the arena)
* QA reporting

## Gait Features

Gait features a generated using https://github.com/KumarLabJax/gaitanalysis
Features generated are described in https://www.cell.com/cell-reports/fulltext/S2211-1247(21)01740-X

## Morphometric features

Morphometric features are generated using https://github.com/KumarLabJax/vFI-features
Features generated are described in https://www.nature.com/articles/s43587-022-00266-0

## JABS features

JABS predicts behavioral events based on pose files. Information contained in pose files is used to generate postural feature vectors. These feature vectors can either be used manually (heuristic classification) or in a supervised manner (supervised classification) to predict behavior. Once behavioral events are predicted by the classifier, events are summarized.
For more details on feature calculation and supervised classification, visit the codebase: https://github.com/KumarLabJax/JABS-behavior-classifier
For more details on heuristic classification and event summarization, visit the codebase: https://github.com/KumarLabJax/JABS-postprocess
Final features presented to the user here are aggregated with [this code](../support_code/behavior_summaries.R).

### Heuristic JABS Classifiers

Heuristic classifiers are calculated using manually defined decision trees on JABS features: https://github.com/KumarLabJax/JABS-postprocess/blob/master/heuristic_classify.py
Available heuristic classifiers are defined here: https://github.com/KumarLabJax/JABS-postprocess/tree/master/heuristic_classifiers

### JABS Behavior Classifiers

JABS classifiers are custom machine learning classifiers trained by a behaviorist.
Briefly, a behaviorist will annotate behavior in video and train a machine learning algorithm to predict on new videos.

## Fecal Boli Counts

Fecal boli are predicted every minute of the video. This pipeline prouces a summary file containing the counts over the time-course of the video.
