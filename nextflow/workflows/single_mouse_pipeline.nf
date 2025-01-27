include { PREDICT_SINGLE_MOUSE_SEGMENTATION; PREDICT_SINGLE_MOUSE_KEYPOINTS } from "./../../nextflow/modules/single_mouse"
include { PREDICT_ARENA_CORNERS } from "./../../nextflow/modules/static_objects"
include { VIDEO_TO_POSE } from "./../../nextflow/modules/utils"

workflow SINGLE_MOUSE_TRACKING {
    take:
    input_video

    main:
    // Generate pose files
    VIDEO_TO_POSE(input_video)
    // Pose v2 is output from this step
    PREDICT_SINGLE_MOUSE_KEYPOINTS(VIDEO_TO_POSE.out)
    PREDICT_SINGLE_MOUSE_SEGMENTATION(PREDICT_SINGLE_MOUSE_KEYPOINTS.out)
    // Completed Pose v6 is output from this step
    PREDICT_ARENA_CORNERS(PREDICT_SINGLE_MOUSE_SEGMENTATION.out)

    emit:
    PREDICT_SINGLE_MOUSE_KEYPOINTS.out
    PREDICT_ARENA_CORNERS.out
}
