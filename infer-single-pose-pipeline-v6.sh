#!/bin/bash
shopt -s extglob

SINGLE_MOUSE_POSE_SCRIPT="run-single-mouse.sh"
USAGE_STR="Usage: ./infer-single-pose-pipeline-v6.sh [-b|--batch movie_list.txt] [-f|--file movie.avi] [-i|--include-v2] [-a|--auto-clip]"

declare -A flags=()
files=() batches=()

while (( $# > 0 )) ; do
  case $1 in
	# Handle short options with bundled arguments
    -[ai][!-]*) set -- "${1:0:2}" "-${1:2}" "${@:2}" ; continue ;;
	# Split short options with arguments
    -[bf]?*) set -- "${1:0:2}" "${1:2}" "${@:2}" ; continue ;;
	# Handle long options with embedded =
    --@(batch|file)=*) set -- "${1%%=*}" "${1#*=}" "${@:2}" ; continue ;;

	#Argument parsing
    -a|--auto-clip) (( flags[a]++ )) ;;
    -i|--include-v2) (( flags[i]++ )) ;;
    -b|--batch) batches+=( "${2?Missing argument for -b|--batch}" ) ; shift ;;
    -f|--file) files+=( "${2?Missing argument for -f|--file}" ) ; shift ;;
    --) shift; break ;;
    -*) printf >&2 'Unknown option %s\n' "$1" "\n${USAGE_STR}" ; exit 1 ;;
    *) break ;;
  esac
  shift
done

# Ignore any remaining arguments silently
# printf 'Unparsed args: %s\n' "${*@Q}"

ERROR_STR="ERROR: you need to provide at least one video file to process.\n${USAGE_STR}"

if [[ ${#files[@]} -eq 0 && ${#batches[@]} -eq 0 ]]; then
  echo "${ERROR_STR}"
  exit 1
fi

# Store all qc from this command in the same file
QC_FILE="/projects/kumar-lab/multimouse-pipeline/qa_logs/single-pose-$(date +"%Y%m%d%H%M").csv"

# Submit array jobs for batch files
for batch in "${batches[@]}"; do
  if [[ ! -f "${batch}" ]]; then
	echo "ERROR: Batch file $batch does not exist."
	continue
  fi
  NUM_VIDEOS=$(wc -l < "${batch}")
  echo "Submitting ${NUM_VIDEOS} videos for single mouse pose in: ${batch}"
  sbatch --export=BATCH_FILE="${batch}",INCLUDE_V2="${flags[i]:=0}",AUTO_CLIP="${flags[a]:=0}",QC_FILE="${QC_FILE}" --array=1-"$NUM_VIDEOS"%56 "${SINGLE_MOUSE_POSE_SCRIPT}"
done

# Submit jobs for individual files
for file in "${files[@]}"; do
  if [[ ! -f "${file}" ]]; then
	echo "ERROR: File $file does not exist, skipping processing."
	continue
  fi
  echo "Submitting single mouse pose for: ${file}"
  sbatch --export=VIDEO_FILE="${file}",INCLUDE_V2="${flags[i]:=0}",AUTO_CLIP="${flags[a]:=0}",QC_FILE="${QC_FILE}" "${SINGLE_MOUSE_POSE_SCRIPT}"
done
