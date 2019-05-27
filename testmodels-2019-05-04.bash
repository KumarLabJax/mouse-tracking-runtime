#!/bin/bash

for (( i=1; i<15; i++ ))
do
    time python tools/testmouseposemodel.py \
        --model-file "model-archive/deep-hres-hyperparams-2019-05-24/output-full-mouse-pose/hdf5mousepose/pose_hrnet/mouse-pose-${i}/model_best.pth" \
        "model-archive/deep-hres-hyperparams-2019-05-24/experiments/hdf5mouse/2019-05-23-param-search/mouse-pose-${i}.yaml"
done

time python tools/testmouseposemodel.py \
    --model-file model-archive/2019-05-16/hyper-param1-output/hdf5mousepose/pose_hrnet/hyper-param1/model_best.pth \
    model-archive/2019-05-16/hyper-param1.yaml
