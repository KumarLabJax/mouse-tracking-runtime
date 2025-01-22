process PREDICT_MULTI_MOUSE_SEGMENTATION {
    label "gpu"
    label "tracking"
    
    input:
    path video_file
    path in_pose

    output:
    path "${video_file.baseName}_seg_data.h5", emit: pose_file

    script:
    """
    cp ${in_pose} "${video_file.baseName}_seg_data.h5"
    python3 ${params.tracking_code_dir}/infer_multi_segmentation.py --video $video_file --out-file "${video_file.baseName}_seg_data.h5"
    """
}

process PREDICT_MULTI_MOUSE_KEYPOINTS {
    label "gpu"
    label "tracking"
    
    input:
    path video_file
    path in_pose

    output:
    path "${video_file.baseName}_pose_est_v3.h5", emit: pose_file

    script:
    """
    cp ${in_pose} "${video_file.baseName}_pose_est_v3.h5"
    python3 ${params.tracking_code_dir}/infer_multi_pose.py --video $video_file --out-file "${video_file.baseName}_pose_est_v3.h5" --batch-size 3
    """
}

process PREDICT_MULTI_MOUSE_IDENTITY {
    label "gpu"
    label "tracking"
    
    input:
    path video_file
    path in_pose

    output:
    path "${video_file.baseName}_pose_est_v3_with_id.h5", emit: pose_file

    script:
    """
    cp ${in_pose} "${video_file.baseName}_pose_est_v3_with_id.h5"
    python3 ${params.tracking_code_dir}/infer_multi_identity.py --video $video_file --out-file "${video_file.baseName}_pose_est_v3_with_id.h5"
    """
}

process GENERATE_MULTI_MOUSE_TRACKLETS {
    label "cpu"
    label "tracking"
    
    input:
    val num_animals
    path in_pose

    output:
    path "${video_file.baseName}_pose_est_v4.h5", emit: pose_file

    // Number of tracklets is not yet a parameter accepted by code, so num_animals is currently ignored
    script:
    """
    cp ${in_pose} "${video_file.baseName}_pose_est_v4.h5"
    python3 ${params.tracking_code_dir}/stitch_tracklets.py --in-pose "${video_file.baseName}_pose_est_v4.h5"
    """
}
