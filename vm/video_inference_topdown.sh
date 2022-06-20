#!/bin/bash
# SBATCH --job-name=topdown_54
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nice
#SBATCH --qos=training

cd /home/ghanba/mousepose_abed/deep-hres-net
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/robson-lab/software/miniconda/lib

~/.conda/envs/mousepose/bin/python tools/inferencevideo_topdown.py  \
        --video_path=/projects/kumar-lab/multimouse-pipeline/model_zoo/videos/B6J_MDB0054.avi         \
        --model_path=/home/ghanba/mousepose_abed/deep-hres-net/output-multi-mouse-topdown-dilated-5-fillholes-alldata/multimousepose/pose_hrnet/multimouse_topdown_1/best_state.pth         \
        --cfg_path=/home/ghanba/mousepose_abed/deep-hres-net/experiments/multimouse/cloudfactory/multimouse_topdown_1.yml         \
        --h5_path=/projects/kumar-lab/multimouse-pipeline/model_zoo/pose_results_cache/B6J_MDB0054.avi_panoptic-deeplab_res50_v1.h5 \
        --h5_path_out=/projects/kumar-lab/multimouse-pipeline/model_zoo/pose_results_cache/B6J_MDB0054.avi_panoptic-deeplab_res50_v1_topdown.h5 
