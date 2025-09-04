/**
 * Predicts multi-mouse segmentation.
 *
 * @param tuple
 *  - video_file The input video file
 *  - in_pose The input pose file
 *
 * @return tuple files
 *  - Path to the original video file.
 *  - Modified pose file with multi-mouse segmentation predicted.
 */
process PREDICT_MULTI_MOUSE_SEGMENTATION {
    label "gpu_long"
    label "tracking"
    label "r_multi_seg"

    input:
    tuple path(video_file), path(in_pose)

    output:
    tuple path(video_file), path("${video_file.baseName}_seg_data.h5"), emit: files

    script:
    """
    cp ${in_pose} "${video_file.baseName}_seg_data.h5"
    mouse-tracking infer multi-segmentation --video $video_file --out-file "${video_file.baseName}_seg_data.h5"
    """
}

/**
 * Predicts multi-mouse keypoints.
 *
 * @param tuple
 *  - video_file The input video file
 *  - in_pose The input pose file
 *
 * @return tuple files
 *  - Path to the original video file.
 *  - Modified pose file with multi-mouse keypoints predicted.
 */
process PREDICT_MULTI_MOUSE_KEYPOINTS {
    label "gpu_long"
    label "tracking"
    label "r_multi_keypoints"

    input:
    tuple path(video_file), path(in_pose)

    output:
    tuple path(video_file), path("${video_file.baseName}_pose_est_v3.h5"), emit: files

    script:
    """
    cp ${in_pose} "${video_file.baseName}_pose_est_v3.h5"
    mouse-tracking infer multi-pose --video $video_file --out-file "${video_file.baseName}_pose_est_v3.h5" --batch-size 3
    """
}

/**
 * Predicts multi-mouse identity.
 *
 * @param tuple
 *  - video_file The input video file
 *  - in_pose The input pose file
 *
 * @return tuple files
 *  - Path to the original video file.
 *  - Modified pose file with multi-mouse identity predicted.
 */
process PREDICT_MULTI_MOUSE_IDENTITY {
    label "gpu"
    label "tracking"
    label "r_multi_identity"

    input:
    tuple path(video_file), path(in_pose)

    output:
    tuple path(video_file), path("${video_file.baseName}_pose_est_v3_with_id.h5"), emit: files

    script:
    """
    cp ${in_pose} "${video_file.baseName}_pose_est_v3_with_id.h5"
    mouse-tracking infer multi-identity --video $video_file --out-file "${video_file.baseName}_pose_est_v3_with_id.h5"
    """
}

/**
 * Generates multi-mouse tracklets from the pose, segmentaiton, and identity data.
 *
 * @param tuple
 *  - video_file The input video file
 *  - in_pose The input pose file
 *  - num_animals The number of animals to generate tracklets for
 *
 * @return tuple files
 *  - Path to the original video file.
 *  - Modified pose file with multi-mouse tracklets generated.
 */
process GENERATE_MULTI_MOUSE_TRACKLETS {
    label "cpu"
    label "tracking"
    label "r_multi_tracklets"

    input:
    tuple path(video_file), path(in_pose)
    val num_animals

    output:
    tuple path(video_file), path("${video_file.baseName}_pose_est_v4.h5"), emit: files

    // Number of tracklets is not yet a parameter accepted by code, so num_animals is currently ignored
    script:
    """
    cp ${in_pose} "${video_file.baseName}_pose_est_v4.h5"
    mouse-tracking utils stitch-tracklets "${video_file.baseName}_pose_est_v4.h5"
    """
}
