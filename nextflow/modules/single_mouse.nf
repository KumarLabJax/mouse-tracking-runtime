process PREDICT_SINGLE_MOUSE_SEGMENTATION {
    label "gpu"
    label "tracking"
    
    input:
    tuple path(video_file), path(in_pose_file)

    output:
    tuple path(video_file), path("${video_file.baseName}_pose_est_v6.h5"), emit: files

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
    tuple path(video_file), path("${video_file.baseName}_pose_est_v2.h5"), emit: files

    script:
    """
    cp ${in_pose_file} "${video_file.baseName}_pose_est_v2.h5"
    python3 ${params.tracking_code_dir}/infer_single_pose.py --video ${video_file} --out-file "${video_file.baseName}_pose_est_v2.h5"
    """
}

process QC_SINGLE_MOUSE {
    label "tracking"

    input:
    tuple path(in_video), path(in_pose_file)
    val(clip_duration)
    val(batch_name)

    output:
    path("${batch_name}_qc.csv"), emit: qc_file

    script:
    """
    for pose_file in ${in_pose_file};
    do
        python3 ${params.tracking_code_dir}/qa_single_pose.py --pose "\${pose_file}" --output "${batch_name}_qc.csv" --duration "${clip_duration}"
    done
    """
}

process CLIP_VIDEO_AND_POSE {
    label "tracking"

    input:
    tuple path(in_video), path(in_pose_file)
    val clip_duration
    
    output:
    tuple path("${in_video.baseName}_trimmed.mp4"), path("${in_pose_file.baseName}_trimmed.h5"), emit: files

    script:
    """
    python3 ${params.tracking_code_dir}/clip_video_to_start.py --in-video "${in_video}" --in-pose "${in_pose_file}" --out-video "${in_video.baseName}_trimmed.mp4" --out-pose "${in_pose_file.baseName}_trimmed.h5" --observation-duration "${clip_duration}" auto
    """
}