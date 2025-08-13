#!/bin/bash

# Usage function
usage() {
    local script_name=$(basename "$0")

    cat << EOF
Usage: $script_name [OPTIONS] [-- NEXTFLOW_ARGS...]

REQUIRED:
    -i, --input BATCH_FILE      Input batch file
    -o, --output OUTPUT_DIR     Output directory

OPTIONAL:
    -n, --pipeline PIPELINE     Nextflow pipeline (default: KumarLabJax/mouse-tracking-runtime)
    -r, --revision REVISION     Pipeline revision/branch/tag (default: none)
    -w, --workflow WORKFLOW     Workflow type (default: single-mouse)
    -p, --profile PROFILE       Nextflow profile (default: sumner2)
    -j, --job-name JOB_NAME     SBATCH job name (default: KL_Tracking_Nextflow)
    -t, --time TIME             Time limit (default: 14-00:00:00)
    -m, --memory MEMORY         Memory allocation (default: 16G)
    --partition PARTITION       SBATCH partition (default: compute)
    --qos QOS                   SBATCH QoS (default: long)
    --resume                    Resume Nextflow run
    --dry-run                   Show what would be submitted without submitting
    -h, --help                  Show this help message

ADDITIONAL NEXTFLOW ARGUMENTS:
    Any arguments after '--' will be passed directly to the nextflow command.
    You can also mix additional nextflow arguments without using '--'.

EXAMPLES:
    $script_name -i my_batch.txt -o results/
    $script_name -i data.txt -o output/ -w multi-mouse -t 7-00:00:00 -m 32G
    $script_name -i batch.txt -o results/ --resume --dry-run
    $script_name -n my-org/my-pipeline -i batch.txt -o results/
    $script_name -n KumarLabJax/mouse-tracking-runtime -r dev -i batch.txt -o results/
    $script_name -r v2.1.0 -i batch.txt -o results/
    $script_name -n ./local-pipeline -i corners_batch.txt -w single-mouse-corrected-corners -o test_output/ --sleap_file manual_correction.slp
    $script_name -i batch.txt -o results/ -- --some_other_param value -nextflow-flag

EOF
}

# Default values
PIPELINE="KumarLabJax/mouse-tracking-runtime"
REVISION=""
WORKFLOW="single-mouse"
PROFILE="sumner2"
JOB_NAME="KL_Tracking_Nextflow"
TIME_LIMIT="14-00:00:00"
MEMORY="16G"
PARTITION="compute"
QOS="long"
RESUME=""
DRY_RUN=false
INPUT_BATCH=""
OUTPUT_DIR=""
ADDITIONAL_ARGS=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--pipeline)
            PIPELINE="$2"
            shift 2
            ;;
        -r|--revision)
            REVISION="$2"
            shift 2
            ;;
        -i|--input)
            INPUT_BATCH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -w|--workflow)
            WORKFLOW="$2"
            shift 2
            ;;
        -p|--profile)
            PROFILE="$2"
            shift 2
            ;;
        -j|--job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        -t|--time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        -m|--memory)
            MEMORY="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --qos)
            QOS="$2"
            shift 2
            ;;
        --resume)
            RESUME="-resume"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            # Everything after -- goes to additional args
            shift
            ADDITIONAL_ARGS+=("$@")
            break
            ;;
        --*)
            # Any other -- argument is likely a nextflow parameter
            ADDITIONAL_ARGS+=("$1")
            if [[ $# -gt 1 && ! "$2" =~ ^- ]]; then
                ADDITIONAL_ARGS+=("$2")
                shift 2
            else
                shift
            fi
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_BATCH" ]]; then
    echo "Error: Input batch file (-i/--input) is required"
    usage
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: Output directory (-o/--output) is required"
    usage
    exit 1
fi

# Check if input file exists
if [[ ! -f "$INPUT_BATCH" ]]; then
    echo "Error: Input batch file '$INPUT_BATCH' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Build the revision flag if specified
REVISION_FLAG=""
if [[ -n "$REVISION" ]]; then
    REVISION_FLAG="-r $REVISION"
fi

# Generate the sbatch script content
SBATCH_SCRIPT=$(cat << EOF
#!/bin/bash

#SBATCH --job-name=$JOB_NAME
#SBATCH -p $PARTITION
#SBATCH -q $QOS
#SBATCH -t $TIME_LIMIT
#SBATCH --mem=$MEMORY
#SBATCH --ntasks=1

# LOAD NEXTFLOW
module use --append /projects/kumar-lab/meta/modules
module load nextflow/stable

# RUN NEXTFLOW PIPELINE
nextflow run $PIPELINE $REVISION_FLAG -profile $PROFILE --input_batch $INPUT_BATCH --workflow $WORKFLOW --pubdir $OUTPUT_DIR $RESUME $(printf " %s" "${ADDITIONAL_ARGS[@]}")
EOF
)

if [[ "$DRY_RUN" == true ]]; then
    echo "=== DRY RUN: Would submit the following sbatch script ==="
    echo "$SBATCH_SCRIPT"
    echo "=============================================="
    echo "Command that would be executed: sbatch <<< \"\$SBATCH_SCRIPT\""
else
    # Submit the job
    echo "Submitting Nextflow pipeline with the following parameters:"
    echo "  Pipeline: $PIPELINE"
    [[ -n "$REVISION" ]] && echo "  Revision: $REVISION"
    echo "  Input batch: $INPUT_BATCH"
    echo "  Output directory: $OUTPUT_DIR"
    echo "  Workflow: $WORKFLOW"
    echo "  Profile: $PROFILE"
    echo "  Job name: $JOB_NAME"
    echo "  Time limit: $TIME_LIMIT"
    echo "  Memory: $MEMORY"
    echo "  Partition: $PARTITION"
    echo "  QoS: $QOS"
    [[ -n "$RESUME" ]] && echo "  Resume: enabled"
    [[ ${#ADDITIONAL_ARGS[@]} -gt 0 ]] && echo "  Additional args: ${ADDITIONAL_ARGS[*]}"
    echo ""

    # Submit using here-string to avoid temporary files
    sbatch <<< "$SBATCH_SCRIPT"
fi