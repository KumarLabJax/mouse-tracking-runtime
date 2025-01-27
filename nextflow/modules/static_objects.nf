process PREDICT_ARENA_CORNERS {
    label "gpu"
    label "tracking"
    
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

process PREDICT_FOOD_HOPPER {
    label "gpu"
    label "tracking"
    
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

process PREDICT_LIXIT {
    label "gpu"
    label "tracking"
    
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
