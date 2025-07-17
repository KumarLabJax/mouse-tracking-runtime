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
    mouse-tracking infer arena-corner --video $video_file --out-file "${video_file.baseName}_with_corners.h5"
    """
}

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
    mouse-tracking infer food-hopper --video $video_file --out-file "${video_file.baseName}_with_food.h5"
    """
}

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
    mouse-tracking infer lixit --video $video_file --out-file "${video_file.baseName}_with_lixit.h5"
    """
}
