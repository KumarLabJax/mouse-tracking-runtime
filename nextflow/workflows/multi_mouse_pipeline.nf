include { PREDICT_MULTI_MOUSE_SEGMENTATION;
          PREDICT_MULTI_MOUSE_KEYPOINTS;
          PREDICT_MULTI_MOUSE_IDENTITY;
          GENERATE_MULTI_MOUSE_TRACKLETS;
 } from "${projectDir}/nextflow/modules/multi_mouse"
include { PREDICT_ARENA_CORNERS;
          PREDICT_FOOD_HOPPER;
          PREDICT_LIXIT;
 } from "${projectDir}/nextflow/modules/static_objects"
include { VIDEO_TO_POSE } from "${projectDir}/nextflow/modules/utils"

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
    pose_v5_lixit = PREDICT_LIXIT(pose_v5_food).files    
}
