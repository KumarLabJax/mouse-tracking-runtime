include { SINGLE_MOUSE_SEGMENTATION; SINGLE_MOUSE_KEYPOINTS } from "./../../nextflow/modules/single_mouse"
include { VIDEO_TO_POSE } from "./../../nextflow/modules/utils"

workflow SINGLE_MOUSE_TRACKING {
    take:
    input_video

    main:
    VIDEO_TO_POSE(input_video)
    SINGLE_MOUSE_KEYPOINTS(input_video, VIDEO_TO_POSE.out.pose_file)
    SINGLE_MOUSE_SEGMENTATION(input_video, SINGLE_MOUSE_KEYPOINTS.out.pose_file)
    // Figure out how to copy the final pose output out
}
