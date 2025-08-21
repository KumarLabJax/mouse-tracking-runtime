#!/bin/bash

#SBATCH --job-name=KL_Tracking_Nextflow
#SBATCH -p compute
#SBATCH -q long
#SBATCH -t 14-00:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1

# LOAD NEXTFLOW
module use --append /projects/kumar-lab/meta/modules
module load nextflow/stable

# RUN TEST PIPELINE
nextflow run KumarLabJax/mouse-tracking-runtime \
 -profile sumner2 \
 --input_batch /projects/kumar-lab/multimouse-pipeline/nextflow-tests/test_batch.txt \
 --workflow single-mouse \
 --pubdir /projects/kumar-lab/multimouse-pipeline/nextflow-test-results/
ß