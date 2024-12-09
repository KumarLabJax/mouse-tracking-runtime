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

# Permanent locations of the singularity images
SINGULARITY_RUNTIME=/projects/kumar-lab/multimouse-pipeline/deployment-runtime-RHEL9-current.sif

ERROR_STR="ERROR: you need to provide a video file to process. Eg: ./infer-single-pose-pipeline-v6.sh /full/path/movie_list.txt [--include-v2]"

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
		H5_V2_OUT_FILE="${FULL_VIDEO_FILE%.*}_pose_est_v2.h5"
		H5_V6_OUT_FILE="${FULL_VIDEO_FILE%.*}_pose_est_v6.h5"
		FAIL_STATE=false

		# Pose V2 Inference step
		echo "Running single mouse pose step:"
		retry singularity exec --nv "${SINGULARITY_RUNTIME}" python3 /kumar_lab_models/mouse-tracking-runtime/infer_single_pose.py --video "${FULL_VIDEO_FILE}" --out-file "${H5_V6_OUT_FILE}" --batch-size 10
		FAIL_STATE=$?

		if [[ ! -z "${INCLUDE_V2}" && "${INCLUDE_V2}" == "true" ]]; then
			cp "${H5_V6_OUT_FILE}" "${H5_V2_OUT_FILE}"
		fi

		# Corner Inference step
		if [[ $FAIL_STATE == 0 ]]; then
			echo "Running arena corner step:"
			retry singularity exec --nv "${SINGULARITY_RUNTIME}" python3 /kumar_lab_models/mouse-tracking-runtime/infer_arena_corner.py --video "${FULL_VIDEO_FILE}" --out-file "${H5_V6_OUT_FILE}"
			FAIL_STATE=$?
		fi

		# Segmentation Inference step
		if [[ $FAIL_STATE == 0 ]]; then
			echo "Running segmentation step:"
			retry singularity exec --nv "${SINGULARITY_RUNTIME}" python3 /kumar_lab_models/mouse-tracking-runtime/infer_single_segmentation.py --video "${FULL_VIDEO_FILE}" --out-file "${H5_V6_OUT_FILE}"
			FAIL_STATE=$?
		fi

		# Fecal Boli Inference step
		if [[ $FAIL_STATE == 0 ]]; then
			echo "Running fecal boli inference step:"
			retry singularity exec --nv "${SINGULARITY_RUNTIME}" python3 /kumar_lab_models/mouse-tracking-runtime/infer_fecal_boli.py --video "${FULL_VIDEO_FILE}" --out-file "${H5_V6_OUT_FILE}"
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
			# should we also output v2?
			if [[ -z "${2}" ]]; then
				INCLUDE_V2="false"
			elif [[ "${2}" == "--include-v2" ]]; then
				INCLUDE_V2="true"
			else
				echo "${ERROR_STR}" >&2
				exit 1
			fi
			# Here we perform a self-submit
			echo "Submitting ${NUM_VIDEOS} videos for single mouse pose in: ${1}"
			sbatch --export=FULL_VIDEO_FILE_LIST="${1}",INCLUDE_V2="${INCLUDE_V2}" --array=1-"$NUM_VIDEOS"%56 "${0}"
	else
			echo "${ERROR_STR}" >&2
			exit 1
	fi
fi
