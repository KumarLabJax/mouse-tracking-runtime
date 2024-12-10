#!/bin/bash
#
#SBATCH --job-name=infer-multimouse-pipeline
#
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_inference
#SBATCH --partition=gpu_a100_mig
#SBATCH --mem=64G
#SBATCH --output=/projects/kumar-lab/multimouse-pipeline/logs/slurm-%x-%A_%a.out

# Permanent locations of the singularity images
SINGULARITY_RUNTIME=/projects/kumar-lab/multimouse-pipeline/deployment-runtime-RHEL9-current.sif

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

# Script is being run by a job in slurm and has been assigned a job ID
if [[ -n "${SLURM_JOB_ID}" ]]; then
	#echo "DUMP OF CURRENT ENVIRONMENT:"
	#env
	FULL_VIDEO_FILE=`head -n $SLURM_ARRAY_TASK_ID $FULL_VIDEO_FILE_LIST | tail -n 1`
	echo "Running on node: ${SLURM_JOB_NODELIST}"
	echo "Assigned GPU: ${CUDA_VISIBLE_DEVICES}"
	echo "Reading from batch: ${FULL_VIDEO_FILE_LIST}"
	echo "Running inference on: ${FULL_VIDEO_FILE}"
	echo "Using the following images:"
	ls -l ${SINGULARITY_RUNTIME}
	echo "Slurm job info: "
	scontrol show job -d ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
	# Force group permissions if default log file used
	LOG_FILE=/projects/kumar-lab/multimouse-pipeline/logs/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
	if [[ -f "${LOG_FILE}" ]]; then
		chmod g+wr ${LOG_FILE}
	fi
	# Actually get to processing
	# Only continue if video file is present
	if [[ -f "${FULL_VIDEO_FILE}" ]]; then
		# Load up required modules
		module load apptainer

		# Setup some useful variables
		H5_V6_OUT_FILE="${FULL_VIDEO_FILE%.*}_pose_est_v6.h5"
		FAIL_STATE=false

		# Panoptic Segmentation Inference step
		echo "Running multi mouse segmentation step:"
		retry singularity exec --nv "${SINGULARITY_RUNTIME}" python3 /kumar_lab_models/mouse-tracking-runtime/infer_multi_segmentation.py --video "${FULL_VIDEO_FILE}" --out-file "${H5_V6_OUT_FILE}"
		FAIL_STATE=$?

		# Topdown Multi-mouse Pose Inference step
		if [[ $FAIL_STATE == 0 ]]; then
			echo "Running topdown multi mouse pose step:"
			retry singularity exec --nv "${SINGULARITY_RUNTIME}" python3 /kumar_lab_models/mouse-tracking-runtime/infer_multi_pose.py --video "${FULL_VIDEO_FILE}" --out-file "${H5_V6_OUT_FILE}" --batch-size 3
			FAIL_STATE=$?
		fi

		# Identity Inference Step
		if [[ $FAIL_STATE == 0 ]]; then
			echo "Running identity step:"
			retry singularity exec --nv "${SINGULARITY_RUNTIME}" python3 /kumar_lab_models/mouse-tracking-runtime/infer_multi_identity.py --model 2023 --video "${FULL_VIDEO_FILE}" --out-file "${H5_V6_OUT_FILE}"
			FAIL_STATE=$?
		fi

		# Tracklet Generation and Stitching Step
		if [[ $FAIL_STATE == 0 ]]; then
			echo "Generating and stitching tracklet step:"
			singularity exec ${SINGULARITY_RUNTIME} python3 /kumar_lab_models/mouse-tracking-runtime/stitch_tracklets.py --in-pose "${H5_V6_OUT_FILE}"
			FAIL_STATE=$?
		fi

		# Corner Inference Step
		if [[ $FAIL_STATE == 0 ]]; then
			echo "Running arena corner step:"
			retry singularity exec --nv "${SINGULARITY_RUNTIME}" python3 /kumar_lab_models/mouse-tracking-runtime/infer_arena_corner.py --video "${FULL_VIDEO_FILE}" --out-file "${H5_V6_OUT_FILE}"
			FAIL_STATE=$?
		fi

		# Food Hopper Inference Step
		if [[ $FAIL_STATE == 0 ]]; then
			echo "Running food hopper step:"
			retry singularity exec --nv "${SINGULARITY_RUNTIME}" python3 /kumar_lab_models/mouse-tracking-runtime/infer_food_hopper.py --video "${FULL_VIDEO_FILE}" --out-file "${H5_V6_OUT_FILE}"
			FAIL_STATE=$?
		fi

		# Lixit Inference Step
		if [[ $FAIL_STATE == 0 ]]; then
			echo "Running lixit step:"
			retry singularity exec --nv "${SINGULARITY_RUNTIME}" python3 /kumar_lab_models/mouse-tracking-runtime/infer_lixit.py --video "${FULL_VIDEO_FILE}" --out-file "${H5_V6_OUT_FILE}"
			FAIL_STATE=$?
		fi

		# Cleanup if successful
		if [[ $FAIL_STATE == 0 ]]; then
			# rm ${FULL_VIDEO_FILE}
			echo "Finished video file: ${FULL_VIDEO_FILE}"
		else
			rm ${H5_V6_OUT_FILE}
			echo "Pipeline failed for Video ${FULL_VIDEO_FILE}, Please Rerun."
		fi
	else
		echo "ERROR: could not find video file: ${FULL_VIDEO_FILE}" >&2
	fi
else
	# the script is being run from command line. We should do a self-submit as an array job
	if [[ -f "${1}" ]]; then
			# echo "${1} is set and not empty"
			NUM_VIDEOS=`wc -l < ${1}`
			# Here we perform a self-submit
			echo "Submitting ${NUM_VIDEOS} Videos with detected number of animals in: ${1}"
			sbatch --export=FULL_VIDEO_FILE_LIST="${1}" --array=1-"$NUM_VIDEOS"%56 "${0}"
	else
			echo "ERROR: you need to provide a video file to process. Eg: ./infer-single-pose-pipeline-v6.sh /full/path/movie_list.txt" >&2
			exit 1
	fi
fi
