#!/bin/bash

for((i=1; i<7; i++))
do
    python -u tools/testmultimouseinference.py \
        --cfg experiments/multimouse/multimouse-${i}.yaml \
        --model-file output-multi-mouse/multimousepose/pose_hrnet/multimouse-${i}/best_state.pth \
        --cvat-files data/multi-mouse/Annotations/*.xml data/multi-mouse/Annotations_NoMarkings/*.xml \
        --image-dir data/multi-mouse/Dataset \
        --image-list data/multi-mouse-val-set.txt \
        --plot-heatmap \
        --dist-out-file output-multi-mouse/dist-out.txt
done

for((i=1; i<11; i++))
do
    python -u tools/testmultimouseinference.py \
        --cfg experiments/multimouse/multimouse_2019-11-19_${i}.yaml \
        --model-file output-multi-mouse/multimousepose/pose_hrnet/multimouse_2019-11-19_${i}/best_state.pth \
        --cvat-files data/multi-mouse/Annotations/*.xml data/multi-mouse/Annotations_NoMarkings/*.xml \
        --image-dir data/multi-mouse/Dataset \
        --image-list data/multi-mouse-val-set.txt \
        --plot-heatmap \
        --dist-out-file output-multi-mouse/dist-out.txt
done

for((i=1; i<9; i++))
do
    python -u tools/testmultimouseinference.py \
        --cfg experiments/multimouse/multimouse_2019-12-19_${i}.yaml \
        --model-file output-multi-mouse/multimousepose/pose_hrnet/multimouse_2019-12-19_${i}/best_state.pth \
        --cvat-files data/multi-mouse/Annotations/*.xml data/multi-mouse/Annotations_NoMarkings/*.xml \
        --image-dir data/multi-mouse/Dataset \
        --image-list data/multi-mouse-val-set.txt \
        --plot-heatmap \
        --dist-out-file output-multi-mouse/dist-out.txt
done
