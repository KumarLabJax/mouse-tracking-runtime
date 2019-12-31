sudo singularity build cuda-2019-05-13.simg docker://nvcr.io/nvidia/cuda:latest
sudo singularity build pytorch_19.04-py3.simg docker://nvcr.io/nvidia/pytorch:19.04-py3


## Point Annotation:

We use CVAT to perform the annotation. To get started follow the Installation Guide at
https://github.com/opencv/cvat. The Users

* Name: MultiPoseEstAndID
* Labels: mouse @checkbox=MarkingUncertain:false @checkbox=LeftShoulderMarked:false @checkbox=RightShoulderMarked:false @checkbox=RearMarked:false
* Image Quality: 95


## Pose Similarity Metrics

* http://cocodataset.org/#keypoints-eval
* https://nanonets.com/blog/human-pose-estimation-2d-guide/
