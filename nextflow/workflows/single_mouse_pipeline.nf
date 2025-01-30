include { PREDICT_SINGLE_MOUSE_SEGMENTATION; PREDICT_SINGLE_MOUSE_KEYPOINTS; CLIP_VIDEO_AND_POSE } from "./../../nextflow/modules/single_mouse"
include { PREDICT_ARENA_CORNERS } from "./../../nextflow/modules/static_objects"
include { VIDEO_TO_POSE } from "./../../nextflow/modules/utils"

workflow SINGLE_MOUSE_TRACKING {
    take:
    input_video

    main:
    // Generate pose files
    // input_video.view()
    pose_init = VIDEO_TO_POSE(input_video).files
    // Pose v2 is output from this step
    pose_v2_data = PREDICT_SINGLE_MOUSE_KEYPOINTS(pose_init).files
    if (params.align_videos) {
        pose_v2_data = CLIP_VIDEO_AND_POSE(pose_v2_data, params.clip_duration).files
    }
    pose_and_seg_data = PREDICT_SINGLE_MOUSE_SEGMENTATION(pose_v2_data).files
    // Completed Pose v6 is output from this step
    pose_v6_data = PREDICT_ARENA_CORNERS(pose_and_seg_data).files

    emit:
    pose_v2_data
    pose_v6_data
}
