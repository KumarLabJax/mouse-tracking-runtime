#!/bin/bash
#
#SBATCH --job-name=train-cloudfactory
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --qos=training
#SBATCH --mem=48G
#SBATCH --nice

cd /home/ghanba/mousepose_abed/deep-hres-net
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/robson-lab/software/miniconda/lib

~/.conda/envs/mousepose/bin/python tools/trainmultimouse.py  \
     --cfg "$1" \
     --data-file /home/ghanba/mousepose_abed/scrap/all3_cloudfactory.h5  \
     --image-dir /projects/compsci/USERS/ghanba/cloudfactory_annotations/all_frames/