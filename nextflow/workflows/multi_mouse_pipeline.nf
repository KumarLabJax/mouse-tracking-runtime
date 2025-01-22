include { MULTI_MOUSE_SEGMENTATION; MULTI_MOUSE_KEYPOINTS; MULTI_MOUSE_IDENTITY; MULTI_MOUSE_TRACKLETS } from "./../../nextflow/modules/multi_mouse"
include { VIDEO_TO_POSE } from "./../../nextflow/modules/utils"

workflow MULTI_MOUSE_TRACKING {
    take:
    input_video
    num_animals

    main:
    VIDEO_TO_POSE(input_video)
    MULTI_MOUSE_SEGMENTATION(input_video, VIDEO_TO_POSE.out.pose_file)
    MULTI_MOUSE_KEYPOINTS(input_video, MULTI_MOUSE_SEGMENTATION.out.pose_file)
    MULTI_MOUSE_IDENTITY(input_video, MULTI_MOUSE_KEYPOINTS.out.pose_file)
    MULTI_MOUSE_TRACKLETS(num_animals, MULTI_MOUSE_IDENTITY.out.pose_file)
    // Figure out how to copy the final pose output out
}
