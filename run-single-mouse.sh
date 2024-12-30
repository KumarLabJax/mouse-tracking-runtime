#!/bin/bash
#
#SBATCH --job-name=infer-singlemouse-pipeline
#
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_inference
#SBATCH --partition=gpu_a100_mig
#SBATCH --mem=16G
#SBATCH --output=/projects/kumar-lab/multimouse-pipeline/logs/slurm-%x-%A_%a.out

# Environment Variables that should be set:
# If running on a single video:
# VIDEO_FILE: The video file to process

# If running on a batch file:
# BATCH_FILE: The batch file to process
# SLURM_ARRAY_TASK_ID: The task ID for the current job (set by SLURM)

# Optional Environment Variables:
# QC_FILE: The file to write the QC results to (automatically generated if not set)
# INCLUDE_V2: Include the V2 pose estimation in the output
# START_FRAME: Manually set the start frame for the video (must include offset)
# AUTO_CLIP: Automatically clip the video to the first detected mouse (will offset 150 frames before first detection)
# CLIP_DURATION: Duration of the video to process (in frames)

# Permanent locations of the singularity images
SINGULARITY_RUNTIME=/projects/kumar-lab/multimouse-pipeline/deployment-runtime-RHEL9-current.sif
CODE_FOLDER=/kumar_lab_models/mouse-tracking-runtime/

# Default clip duration to 1hr of frames if not set
if [[ -z "${CLIP_DURATION}" ]]; then
  CLIP_DURATION=108000
fi

# Basic function that retries a command up to 5 times
function retry {
	local n=1
	local max=5
	while true; do
		"$@" && break || {
			if [[ $n -lt $max ]]; then
				((n++))
				echo "Command failed. Attempt $n/$max:"
			else
				echo "The command has failed after $n attempts." >&2
				return 1
			fi
		}
	done
}

# Print logging of the assigned resources
echo "Running on node: ${SLURM_JOB_NODELIST}"
echo "Assigned GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Using the following images:"
ls -l ${SINGULARITY_RUNTIME}
echo "Slurm job info: "
# Job string is different for array jobs
if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]; then
	JOB_STRING="${SLURM_JOB_ID}"
else
	JOB_STRING="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
scontrol show job -d ${JOB_STRING}

# Force group permissions if default log file used
LOG_FILE=/projects/kumar-lab/multimouse-pipeline/logs/slurm-${SLURM_JOB_NAME}-${JOB_STRING}.out
if [[ -f "${LOG_FILE}" ]]; then
	chmod g+wr ${LOG_FILE}
fi

# If QC_FILE is not set, use a default location
# Group permissions on QC file are changed later, to ensure header is written
if [[ -z "${QC_FILE}" ]]; then
  QC_FILE=/projects/kumar-lab/multimouse-pipeline/qa_logs/single-pose-${JOB_STRING}.csv
fi

# Check if we are running a batch job or a single job and assign the video file
if [[ -z "${VIDEO_FILE}" ]]; then
	# VIDEO_FILE is not set, so we are running an array job
	if [[ -z "${BATCH_FILE}" || -z "${SLURM_ARRAY_TASK_ID}" ]]; then
		echo "Video batch file or slurm array task ID not found, exiting."
		exit 1
	fi
	echo "Reading from batch: ${BATCH_FILE}"
	VIDEO_FILE=`head -n $SLURM_ARRAY_TASK_ID $BATCH_FILE | tail -n 1`
fi

echo "Processing video file: ${VIDEO_FILE}"
# Only continue if video file is present
if [[ -z "${VIDEO_FILE}" ]]; then
	echo "No video file found for this task, exiting."
	exit 1
fi

# Load up required modules
module load apptainer

# Setup some useful variables
H5_V2_OUT_FILE="${VIDEO_FILE%.*}_pose_est_v2.h5"
H5_V6_OUT_FILE="${VIDEO_FILE%.*}_pose_est_v6.h5"

# Cleanup process to run when any step fails
function fail_cleanup() {
  rm ${H5_V6_OUT_FILE}
  if [[ -f "${H5_V2_OUT_FILE}" ]]; then
    rm ${H5_V2_OUT_FILE}
  fi
  # Was the video trimmed? If so, remove it
  if [[ "${VIDEO_FILE##*_trimmed}" = ".mp4" ]]; then
	  rm ${VIDEO_FILE}
  fi
  echo "Error on pipeline script line $1"
  echo $( head -n $1 ${0} | tail -n 1 )
  echo "Pipeline failed for Video ${VIDEO_FILE}, Please Rerun."
  exit 1
}

trap 'fail_cleanup $LINENO' ERR

#----------------------------------------------
# Run the pipeline
#----------------------------------------------

# Pose V2 Inference step
echo "Running single mouse pose step:"
retry singularity exec --nv "${SINGULARITY_RUNTIME}" python3 "${CODE_FOLDER}infer_single_pose.py" --video "${VIDEO_FILE}" --out-file "${H5_V6_OUT_FILE}" --batch-size 10

# Clip the video file if requested
# Manual clip if start frame is provided
if [[ ! -z "${START_FRAME}" ]]; then
  echo "Clipping video file manually to frame ${START_FRAME}"
  singularity exec "${SINGULARITY_RUNTIME}" python3 "${CODE_FOLDER}clip_video_to_start.py" --in-video "${VIDEO_FILE}" --in-pose "${H5_V6_OUT_FILE}" --out-video "${VIDEO_FILE%.*}_trimmed.mp4" --out-pose "${VIDEO_FILE%.*}_trimmed_pose_est_v6.h5" --observation-duration "${CLIP_DURATION}" manual --start-frame "${START_FRAME}"
  # Reassign processing to the trimmed video file
  rm ${H5_V6_OUT_FILE}
  VIDEO_FILE="${VIDEO_FILE%.*}_trimmed.mp4"
  H5_V2_OUT_FILE="${VIDEO_FILE%.*}_pose_est_v2.h5"
  H5_V6_OUT_FILE="${VIDEO_FILE%.*}_pose_est_v6.h5"
# Auto clip if requested
elif [[ ! -z "${AUTO_CLIP}" && "${AUTO_CLIP}" -eq 1 ]]; then
  echo "Auto-clipping video file"
  singularity exec "${SINGULARITY_RUNTIME}" python3 "${CODE_FOLDER}clip_video_to_start.py" --in-video "${VIDEO_FILE}" --in-pose "${H5_V6_OUT_FILE}" --out-video "${VIDEO_FILE%.*}_trimmed.mp4" --out-pose "${VIDEO_FILE%.*}_trimmed_pose_est_v6.h5" --observation-duration "${CLIP_DURATION}" auto
  # Reassign processing to the trimmed video file
  rm ${H5_V6_OUT_FILE}
  VIDEO_FILE="${VIDEO_FILE%.*}_trimmed.mp4"
  H5_V2_OUT_FILE="${VIDEO_FILE%.*}_pose_est_v2.h5"
  H5_V6_OUT_FILE="${VIDEO_FILE%.*}_pose_est_v6.h5"
fi

# Save copy of V2 output if requested
if [[ ! -z "${INCLUDE_V2}" && "${INCLUDE_V2}" -eq 1 ]]; then
	cp "${H5_V6_OUT_FILE}" "${H5_V2_OUT_FILE}"
fi

# Corner Inference step
echo "Running arena corner step:"
retry singularity exec --nv "${SINGULARITY_RUNTIME}" python3 "${CODE_FOLDER}infer_arena_corner.py" --video "${VIDEO_FILE}" --out-file "${H5_V6_OUT_FILE}"

# Segmentation Inference step
echo "Running segmentation step:"
retry singularity exec --nv "${SINGULARITY_RUNTIME}" python3 "${CODE_FOLDER}infer_single_segmentation.py" --video "${VIDEO_FILE}" --out-file "${H5_V6_OUT_FILE}"

# Fecal Boli Inference step
echo "Running fecal boli inference step:"
retry singularity exec --nv "${SINGULARITY_RUNTIME}" python3 "${CODE_FOLDER}infer_fecal_boli.py" --video "${VIDEO_FILE}" --out-file "${H5_V6_OUT_FILE}"

# Run QC Step
echo "Running QC step:"
singularity exec "${SINGULARITY_RUNTIME}" python3 "${CODE_FOLDER}qa_single_pose.py" --pose "${H5_V6_OUT_FILE}" --output "${QC_FILE}" --duration "${CLIP_DURATION}"
# Force group permissions on qc file
if [[ -f "${QC_FILE}" ]]; then
	chmod g+wr ${QC_FILE}
fi

# Cleanup if successful
# rm ${VIDEO_FILE}
echo "Finished video file: ${VIDEO_FILE}"
