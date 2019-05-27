sudo singularity build cuda-2019-05-13.simg docker://nvcr.io/nvidia/cuda:latest
sudo singularity build pytorch_19.04-py3.simg docker://nvcr.io/nvidia/pytorch:19.04-py3


Model #6 seems to be highest quality:

=> loading configuration from model-archive/deep-hres-hyperparams-2019-05-24/experiments/hdf5mouse/2019-05-23-param-search/mouse-pose-6.yaml
=> loading model from model-archive/deep-hres-hyperparams-2019-05-24/output-full-mouse-pose/hdf5mousepose/pose_hrnet/mouse-pose-6/model_best.pth
L2 Pixel Error Mean of Means:   3.5032272 359.56085
NOSE Pixel Error:               3.5473418 Max: 359.56085
LEFT_EAR Pixel Error:           3.1541655 Max: 349.42523
RIGHT_EAR Pixel Error:          2.5225682 Max: 19.235384
BASE_NECK Pixel Error:          2.5294154 Max: 15.811388
LEFT_FRONT_PAW Pixel Error:     4.5933537 Max: 19.79899
RIGHT_FRONT_PAW Pixel Error:    4.366912 Max: 19.209373
CENTER_SPINE Pixel Error:       3.8919368 Max: 15.652476
LEFT_REAR_PAW Pixel Error:      5.118787 Max: 27.45906
RIGHT_REAR_PAW Pixel Error:     3.9783306 Max: 24.5153
BASE_TAIL Pixel Error:          2.7562897 Max: 29.0
MID_TAIL Pixel Error:           3.3737302 Max: 21.023796
TIP_TAIL Pixel Error:           2.2058926 Max: 35.805027
