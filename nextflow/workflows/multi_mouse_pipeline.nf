include { PREDICT_MULTI_MOUSE_SEGMENTATION;
          PREDICT_MULTI_MOUSE_KEYPOINTS;
          PREDICT_MULTI_MOUSE_IDENTITY;
          GENERATE_MULTI_MOUSE_TRACKLETS;
 } from "${projectDir}/nextflow/modules/multi_mouse"
include { PREDICT_ARENA_CORNERS;
          PREDICT_FOOD_HOPPER;
          PREDICT_LIXIT;
 } from "${projectDir}/nextflow/modules/static_objects"
include { VIDEO_TO_POSE;
          PUBLISH_RESULT_FILE as PUBLISH_MM_POSE_V6;
 } from "${projectDir}/nextflow/modules/utils"

workflow MULTI_MOUSE_TRACKING {
    take:
    input_video
    num_animals

    main:
    pose_init = VIDEO_TO_POSE(input_video).files
    pose_seg_only = PREDICT_MULTI_MOUSE_SEGMENTATION(pose_init).files
    pose_v3 = PREDICT_MULTI_MOUSE_KEYPOINTS(pose_seg_only).files
    pose_v4_no_tracks = PREDICT_MULTI_MOUSE_IDENTITY(pose_v3).files
    pose_v4 = GENERATE_MULTI_MOUSE_TRACKLETS(pose_v4_no_tracks, num_animals).files
    pose_v5_arena = PREDICT_ARENA_CORNERS(pose_v4).files
    pose_v5_food = PREDICT_FOOD_HOPPER(pose_v5_arena).files
    // While this is a pose_v5 step, segmentation (v6) was already done as the first step
    pose_v6 = PREDICT_LIXIT(pose_v5_food).files

    // Publish the pose v6 results
    v_6_poses_renamed = pose_v6.map { video, pose ->
        tuple(pose, "results/${video.baseName.replace("%20", "/")}_pose_est_v6.h5")
    }
    PUBLISH_MM_POSE_V6(v_6_poses_renamed)

    emit:
    pose_v6
}
