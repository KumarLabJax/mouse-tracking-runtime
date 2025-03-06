#!/bin/bash

#SBATCH --job-name=KL_Tracking_Nextflow
#SBATCH -p compute
#SBATCH -q long
#SBATCH -t 14-00:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1

cd /projects/kumar-lab/multimouse-pipeline/nextflow-code/

# LOAD NEXTFLOW
module use --append /projects/omics_share/meta/modules
module load nextflow/24.04.4

# RUN TEST PIPELINE
nextflow run main.nf \
 -c nextflow.config \
 -c nextflow/configs/profiles/sumner2.config \
 --input_batch /projects/kumar-lab/multimouse-pipeline/nextflow-tests/test_batch.txt \
 --workflow single-mouse \
 --pubdir /projects/kumar-lab/multimouse-pipeline/nextflow-test-results/ \