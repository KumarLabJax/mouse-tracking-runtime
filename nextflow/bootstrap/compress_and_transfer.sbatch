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
nextflow run KumarLabJax/mouse-tracking-runtime \
    -r compression-standalone \
    -main-script bootstrap/compress_dropbox_to_cloudian.nf \
    -profile compression \
    --input_batch TEST_HPC.txt \
    -with-trace \
    -with-report \
    -latest
