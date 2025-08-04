include { PREDICT_SINGLE_MOUSE_SEGMENTATION; PREDICT_SINGLE_MOUSE_KEYPOINTS; CLIP_VIDEO_AND_POSE } from "${projectDir}/nextflow/modules/single_mouse"
include { PREDICT_ARENA_CORNERS } from "${projectDir}/nextflow/modules/static_objects"
include { PREDICT_FECAL_BOLI } from "${projectDir}/nextflow/modules/fecal_boli"
include { QC_SINGLE_MOUSE } from "${projectDir}/nextflow/modules/single_mouse"
include { VIDEO_TO_POSE;
          GET_WORKFLOW_VERSION;
          SELECT_COLUMNS;
          PUBLISH_RESULT_FILE as PUBLISH_SM_QC;
          REMOVE_URLIFY_FIELDS as NOURL_QC;
          ADD_COLUMN as ADD_VERSION_QC;
          PUBLISH_RESULT_FILE as PUBLISH_SM_TRIMMED_VID;
          PUBLISH_RESULT_FILE as PUBLISH_SM_POSE_V2;
          PUBLISH_RESULT_FILE as PUBLISH_SM_POSE_V6;
          PUBLISH_RESULT_FILE as PUBLISH_SM_POSE_V6_NOCORN;
 } from "${projectDir}/nextflow/modules/utils"

workflow SINGLE_MOUSE_TRACKING {
    take:
    input_video

    main:
    // Generate pose files
    pose_init = VIDEO_TO_POSE(input_video).files
    // Pose v2 is output from this step
    pose_v2_data = PREDICT_SINGLE_MOUSE_KEYPOINTS(pose_init).files
    if (params.align_videos) {
        pose_v2_data = CLIP_VIDEO_AND_POSE(pose_v2_data, params.clip_duration).files
    }
    pose_and_seg_data = PREDICT_SINGLE_MOUSE_SEGMENTATION(pose_v2_data).files
    // Completed Pose v6 is output from this step
    pose_with_corners = PREDICT_ARENA_CORNERS(pose_and_seg_data).files
    pose_v6_data = PREDICT_FECAL_BOLI(pose_with_corners).files

    // Publish the pose v2 results
    trimmed_video_files = pose_v2_data.map { video, pose ->
        tuple(video, "results/${video.name.replace("%20", "/")}")
    }
    PUBLISH_SM_TRIMMED_VID(trimmed_video_files)
    v2_poses_renamed = pose_v2_data.map { video, pose ->
        tuple(pose, "results/${video.baseName.replace("%20", "/")}_pose_est_v2.h5")
    }
    PUBLISH_SM_POSE_V2(v2_poses_renamed)

    emit:
    pose_v2_data
    pose_v6_data
}

workflow SPLIT_BY_CORNERS {
    take:
    input_pose_v6_batch

    main:
    // QC the results
    input_poses = input_pose_v6_batch.map { video, pose -> pose }.collect()
    QC_SINGLE_MOUSE(input_poses, params.clip_duration, params.batch_name)
    qc_output = QC_SINGLE_MOUSE.out.qc_file
    workflow_version = GET_WORKFLOW_VERSION().version
    PUBLISH_SM_QC(NOURL_QC(ADD_VERSION_QC(qc_output, "nextflow_version", workflow_version)).map { file -> tuple(file, "qc_${params.batch_name}.csv") })

    // Split based on corners being present
    joined_channel = SELECT_COLUMNS(QC_SINGLE_MOUSE.out, 'pose_file', 'corners_present')
        .splitCsv(header: true, sep: ',')
        .map(row -> [row.pose_file, row.corners_present])
    // Split qc filenames into present and missing
    split_channel = joined_channel.branch { v, c ->
        present: c.contains("True")
            return v
        missing: c.contains("False")
            return v
    }
    // Split path channel with defaults
    // Note that the `.val` is necessary for branch to recognize it as a conditional
    // rather than a groovy statement
    present_filenames = split_channel.present.ifEmpty("INVALID_POSE_FILE").toList()
    missing_filenames = split_channel.missing.ifEmpty("INVALID_POSE_FILE").toList()
    branched = input_pose_v6_batch.branch { video, pose ->
        present: present_filenames.val.contains(pose.getName().toString())
            return tuple(video, pose)
        missing: missing_filenames.val.contains(pose.getName().toString())
            return tuple(video, pose)
    }
    v6_with_corners = branched.present.ifEmpty(tuple(params.default_feature_input[0], params.default_feature_input[1]))
    v6_without_corners = branched.missing.ifEmpty(tuple(params.default_manual_correction_input[0], params.default_manual_correction_input[1]))

    // Publish the pose files
    v6_poses_renamed = v6_with_corners.map { video, pose ->
        tuple(pose, "results/${video.baseName.replace("%20", "/")}_pose_est_v6.h5")
    }
    PUBLISH_SM_POSE_V6(v6_poses_renamed)
    // Corners that failed are placed in a separate folder with url-ified names
    v6_no_corners_renamed = v6_without_corners.map { video, pose ->
        tuple(pose, "failed_corners/${file(video).baseName}_pose_est_v6.h5")
    }
    PUBLISH_SM_POSE_V6_NOCORN(v6_no_corners_renamed)

    emit:
    v6_with_corners
    v6_without_corners
}
