/**
 * Predicts arena corners.
 *
 * @param tuple
 *  - video_file Path to the video file.
 *  - in_pose Path to the input pose file.
 *
 * @return tuple files
 *  - Path to the original video file.
 *  - Modified pose file with arena corners predicted.
 */
process PREDICT_ARENA_CORNERS {
    label "gpu"
    label "tracking"
    label "r_arena_corners"
    
    input:
    tuple path(video_file), path(in_pose)

    output:
    tuple path(video_file), path("${video_file.baseName}_with_corners.h5"), emit: files

    script:
    """
    cp ${in_pose} "${video_file.baseName}_with_corners.h5"
    python3 ${params.tracking_code_dir}/infer_arena_corner.py --video $video_file --out-file "${video_file.baseName}_with_corners.h5"
    """
}

/*
 * Predicts food hopper position.
 *
 * @param tuple
 *  - video_file Path to the video file.
 *  - in_pose Path to the input pose file.
 *
 * @return tuple files
 *  - Path to the original video file.
 *  - Modified pose file with food hopper predicted.
 */
process PREDICT_FOOD_HOPPER {
    label "gpu"
    label "tracking"
    label "r_food_hopper"
    
    input:
    tuple path(video_file), path(in_pose)

    output:
    tuple path(video_file), path("${video_file.baseName}_with_food.h5"), emit: files

    script:
    """
    cp ${in_pose} "${video_file.baseName}_with_food.h5"
    python3 ${params.tracking_code_dir}/infer_food_hopper.py --video $video_file --out-file "${video_file.baseName}_with_food.h5"
    """
}

/**
 * Predicts lixit position.
 *
 * @param tuple
 *  - video_file Path to the video file.
 *  - in_pose Path to the input pose file.
 *
 * @return tuple files
 *  - Path to the original video file.
 *  - Modified pose file with lixit predicted.
 */
process PREDICT_LIXIT {
    label "gpu"
    label "tracking"
    label "r_lixit"
    
    input:
    tuple path(video_file), path(in_pose)

    output:
    tuple path(video_file), path("${video_file.baseName}_with_lixit.h5"), emit: files

    script:
    """
    cp ${in_pose} "${video_file.baseName}_with_lixit.h5"
    python3 ${params.tracking_code_dir}/infer_lixit.py --video $video_file --out-file "${video_file.baseName}_with_lixit.h5"
    """
}
