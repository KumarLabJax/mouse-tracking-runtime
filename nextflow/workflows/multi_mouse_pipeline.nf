/**
 * This module contains the multi-mouse tracking pipeline.
 * It processes video input to track multiple mice, predict their poses, and identify static objects.
 */
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
          PUBLISH_RESULT_FILE as PUBLISH_COMPRESSED_VIDEO;
 } from "${projectDir}/nextflow/modules/utils"
 include { COMPRESS_VIDEO_CRF } from "${projectDir}/nextflow/modules/compression"

/**
 * Main workflow for predicting multimouse poses from video input.
 *
 * @param path input_video The input video file to process.
 * @param val num_animals The number of animals to track in the video.
 *
 * @return tuple pose_v6
 *  - Path to the input video file.
 *  - Path to the pose v6 file produced.
 *
 * @publish ./results/ Pose v6 results
 */
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

    // Compress the original video
    compressed_videos = COMPRESS_VIDEO_CRF(input_video.combine([23]).combine([3000]))
    compressed_renamed = compressed_videos.map { video -> 
        tuple(video, "results/${video.baseName.replace("_g3000_crf23", "").replace("%20", "/")}_compressed.mp4")
    }
    PUBLISH_COMPRESSED_VIDEO(compressed_renamed)

    emit:
    pose_v6
}
