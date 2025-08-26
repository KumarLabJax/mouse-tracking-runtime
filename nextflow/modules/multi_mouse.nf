process PREDICT_MULTI_MOUSE_SEGMENTATION {
    label "gpu"
    label "tracking"
    
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

process PREDICT_MULTI_MOUSE_KEYPOINTS {
    label "gpu"
    label "tracking"
    
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

process PREDICT_MULTI_MOUSE_IDENTITY {
    label "gpu"
    label "tracking"
    
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

process GENERATE_MULTI_MOUSE_TRACKLETS {
    label "cpu"
    label "tracking"
    
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
