/**
 * This module contains process definitions for fecal boli prediction and feature calculations.
 */

 /**
 * Predicts fecal boli in the video at minute intervals.
 *
 * @param tuple
 *  - video_file The input video file.
 *  - in_pose The input pose file.
 *
 * @return tuple files
 *  - Path to the original video file.
 *  - Modified pose file with fecal boli predictions.
 */
process PREDICT_FECAL_BOLI {
    label "gpu"
    label "tracking"
    label "r_fboli_predict"

    input:
    tuple path(video_file), path(in_pose)

    output:
    tuple path(video_file), path("${video_file.baseName}_with_fecal_boli.h5"), emit: files

    script:
    """
    cp ${in_pose} "${video_file.baseName}_with_fecal_boli.h5"
    mouse-tracking infer fecal-boli --video ${video_file} --out-file "${video_file.baseName}_with_fecal_boli.h5" --frame-interval 1800
    """
}

/**
 * Extracts fecal boli features from the pose data.
 *
 * @param tuple
 *  - video_file The input video file.
 *  - in_pose The input pose file.
 *
 * @return fecal_boli The generated fecal boli feature CSV file.
 */
process EXTRACT_FECAL_BOLI_BINS {
    label "tracking"
    label "r_fboli_extract"

    input:
    tuple path(video_file), path(in_pose)

    output:
    path("${video_file.baseName}_fecal_boli.csv"), emit: fecal_boli

    script:
    """
    if [ ! -f "${video_file.baseName}_pose_est_v6.h5" ]; then
        ln -s ${in_pose} "${video_file.baseName}_pose_est_v6.h5"
    fi
    mouse-tracking utils aggregate-fecal-boli . --folder-depth 0 --num-bins ${params.clip_duration.intdiv(1800)} --output ${video_file.baseName}_fecal_boli.csv
    """
}