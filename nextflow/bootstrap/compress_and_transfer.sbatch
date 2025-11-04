#!/bin/bash
#
#SBATCH --job-name=KL_Compress_And_Transfer
#
#SBATCH --qos=long
#SBATCH --time=4-0:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# 25TB of data chunks should ideally have jobs run for about 36hrs, so we request 4 days just to be safe.
# 
# 2800 [files in largest batch] * (
#  1.5 [minutes retrieval] / 2 [simultaneous jobs] +
#  ( 24 [minutes compression] / 250 [simultaneous jobs] ) * ( 2/3 [load efficiency] )
# ) / 60 [minutes per hour]

module load nextflow

# There are batch files in the folder containing data to process.
# Batch files are titled batch_#.txt
BATCH_START=1
BATCH_END=73
cd /projects/kumar-lab/bgeuther/dropbox-compression/dropbox_file_lists/batches_25TB/

# Sanity checks on environment variable pointing to a valid batch to process.
if ! [[ $CURRENT_BATCH =~ '^[0-9]+$' ]]; then
   echo "Error: CURRENT_BATCH is unset or not a number. Current value: \"${CURRENT_BATCH}\"" >&2
   echo "Example submission: \"sbatch ${0} --export=CURRENT_BATCH=1\"" >&2
   exit 1
elif ! [[ -f batch_${CURRENT_BATCH}.txt ]]; then
    echo "Error: batch_${CURRENT_BATCH}.txt does not exist in the working directory. Cannot process." >&2
    exit 1
fi

# Run the actual processing
nextflow run KumarLabJax/mouse-tracking-runtime \
    -r compression-standalone \
    -main-script bootstrap/compress_dropbox_to_cloudian.nf \
    -profile compression \
    --input_batch batch_${CURRENT_BATCH}.txt \
    -with-trace \
    -with-report \
    -latest

# Exit if something went wrong
STATE=$?
if [[ $STATE == 1 ]]; then
    echo "Error: Nextflow run on batch ${CURRENT_BATCH} failed." >&2
    exit 1
else
    echo "Nextflow run on batch ${CURRENT_BATCH} succeeded.
fi

# Submit next batch if there is one.
NEXT_BATCH=$((CURRENT_BATCH + 1))
if [[ $NEXT_BATCH <= $BATCH_END ]]; then
    # Self-submit the next batch
    SUBMITTED_JOBID=$(sbatch --parsable --export=CURRENT_BATCH="${NEXT_BATCH}" "${0}")
    echo "Submitted next processing job (batch_${NEXT_BATCH}). Job ID: ${SUBMITTED_JOBID}.
else
    echo "Batches complete! Exiting self-submitting loop.
fi
