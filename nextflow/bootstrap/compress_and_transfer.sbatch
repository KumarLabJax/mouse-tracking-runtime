#!/bin/bash
#
#SBATCH --job-name=KL_Compress_And_Transfer
#
#SBATCH --qos=long
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G

module load nextflow

cd /projects/kumar-lab/bgeuther/dropbox-compression
nextflow run compress_dropbox_to_cloudian.nf -c dropbox_to_cloudian.config --input_batch TEST_HPC.txt -with-trace -with-report
