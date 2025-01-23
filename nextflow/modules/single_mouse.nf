process PREDICT_SINGLE_MOUSE_SEGMENTATION {
    label "gpu"
    label "tracking"
    
    input:
    tuple path(video_file), path(in_pose_file)

    output:
    tuple path(video_file), path("${video_file.baseName}_pose_est_v6.h5"), emit: pose_file

    script:
    """
    cp ${in_pose_file} "${video_file.baseName}_pose_est_v6.h5"
    python3 ${params.tracking_code_dir}/infer_single_segmentation.py --video ${video_file} --out-file "${video_file.baseName}_pose_est_v6.h5"
    """
}

process PREDICT_SINGLE_MOUSE_KEYPOINTS {
    label "gpu"
    label "tracking"
    
    input:
    tuple path(video_file), path(in_pose_file)

    output:
    tuple path(video_file), path("${video_file.baseName}_pose_est_v2.h5"), emit: pose_file

    script:
    """
    cp ${in_pose_file} "${video_file.baseName}_pose_est_v2.h5"
    python3 ${params.tracking_code_dir}/infer_single_pose.py --video ${video_file} --out-file "${video_file.baseName}_pose_est_v2.h5"
    """
}
