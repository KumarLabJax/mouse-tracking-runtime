# Deployment Runtime Pipelines

This is a collection of Kumar Lab pipelines converted over to a flexible deployment runtime.
This is specifically NOT designed for training new models, but rather takes exported/frozen models and runs inference on videos using them.

This repository uses both Pytorch and Tensorflow Serving (TFS).

# Installation

Both Google Colab and singularity environments are supported. This environment is used because it is a convenient method to have both pytorch and tensorflow present.

## Singularity Containers

See the [container definition file](vm/deployment-runtime-RHEL9.def) in the vm folder. This container is based off a google colab public docker.

# Available Models

See [model docs](docs/models.md) for information about available models.

# Running a pipeline

Pipelines are run using nextflow. For a list of all available parameters, see [nextflow parameters](nextflow.config). Not all parameters will affect all pipeline workflows.

If you are running the pipeline on the HPC, you should submit the workflow as a job. You can use the example test as a starting template: [test-nextflow.sh](test-nextflow.sh).

Input files should include the full path. An easy way to generate the list of inputs for `input_batch` is to run `find $(pwd) -name '*.avi' > video_batch.txt`.

## Single Mouse Pipelines

See [docs/pipelines.md](docs/pipelines.md) for more specific information about the structure of the pipeline.

### Video to Features

The nextflow workflow `single-mouse` generates feature tables from input video.

Input:
* Video Files

Output:
* `workflow_version.txt` information related to the specific workflow run.
* Folder named `results` with clipped videos, pose_v2 predictions, and pose_v6 predictions with corners.
* Folder named `failed_corners` with pose_v6 predictions that failed corners.
* `manual_corner_corrections.slp` sleap file containing frames to manually correct corners.
* `qc_batch_[date].csv` QC file reporting single mouse pose quality metrics.
* pose_v2 related features
 * `gait.csv` feature file containing gait pipeline features.
 * `morphometrics.csv` feature file containing morphometric features.
* pose_v6 related features (successful corners only)
 * `features.csv` feature file containing JABS-related features.
 * `fecal_boli.csv` prediction file containing fecal boli counts for each video, used in growth curve modeling.

Example Command:
`nextflow -c nextflow.config -c nextflow/configs/profiles/development.config run main.nf --input_batch /path/to/video_batch.txt --workflow single-mouse --pubdir /path/to/output_folder`

### Corner Correction to Features

The nextflow workflow `single-mouse-corrected-corners` completes the `single-mouse` pipeline for files that required their corners to be manual correction.

Input:
* Corrected Sleap file
* Folder containing pose_v6 predictions to add corners

Output:
* `workflow_version.txt` information related to the specific workflow run.
* pose_v6 related features
 * `features.csv` feature file containing JABS-related features.
 * `fecal_boli.csv` prediction file containing fecal boli counts for each video, used in growth curve modeling.

Example Command:
`nextflow -c nextflow.config -c nextflow/configs/profiles/development.config run main.nf --input_batch /path/to/pose_v6_batch.txt --sleap_file /path/to/corner-correction.slp --workflow single-mouse-corrected-corners --pubdir /path/to/output_folder`

### Pose File (v6) to Features

The nextflow workflow `single-mouse-v6-features` generates pose_v6 features from pose files.

Input:
* Pose files (arena corners required!)

Output:
* `workflow_version.txt` information related to the specific workflow run.
* pose_v6 related features
 * `features.csv` feature file containing JABS-related features.
 * `fecal_boli.csv` prediction file containing fecal boli counts for each video, used in growth curve modeling.

Example Command:
`nextflow -c nextflow.config -c nextflow/configs/profiles/development.config run main.nf --input_batch /path/to/pose_v6_batch.txt --workflow single-mouse-v6-features --pubdir /path/to/output_folder`
