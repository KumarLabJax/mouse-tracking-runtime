include { PREDICT_SINGLE_MOUSE_SEGMENTATION; PREDICT_SINGLE_MOUSE_KEYPOINTS } from "./../../nextflow/modules/single_mouse"
include { PREDICT_ARENA_CORNERS } from "./../../nextflow/modules/static_objects"
include { VIDEO_TO_POSE } from "./../../nextflow/modules/utils"

workflow SINGLE_MOUSE_TRACKING {
    take:
    input_video

    main:
    VIDEO_TO_POSE(input_video)
    PREDICT_SINGLE_MOUSE_KEYPOINTS(input_video, VIDEO_TO_POSE.out.pose_file)
    PREDICT_SINGLE_MOUSE_SEGMENTATION(input_video, PREDICT_SINGLE_MOUSE_KEYPOINTS.out.pose_file)
    PREDICT_ARENA_CORNERS(input_video, PREDICT_SINGLE_MOUSE_SEGMENTATION.out.pose_file)
    // Figure out how to copy the final pose output out
}
