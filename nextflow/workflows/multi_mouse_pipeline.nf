include { PREDICT_MULTI_MOUSE_SEGMENTATION; PREDICT_MULTI_MOUSE_KEYPOINTS; PREDICT_MULTI_MOUSE_IDENTITY; GENERATE_MULTI_MOUSE_TRACKLETS } from "./../../nextflow/modules/multi_mouse"
include { VIDEO_TO_POSE } from "./../../nextflow/modules/utils"

workflow MULTI_MOUSE_TRACKING {
    take:
    input_video
    num_animals

    main:
    VIDEO_TO_POSE(input_video)
    PREDICT_MULTI_MOUSE_SEGMENTATION(input_video, VIDEO_TO_POSE.out.pose_file)
    PREDICT_MULTI_MOUSE_KEYPOINTS(input_video, PREDICT_MULTI_MOUSE_SEGMENTATION.out.pose_file)
    PREDICT_MULTI_MOUSE_IDENTITY(input_video, PREDICT_MULTI_MOUSE_KEYPOINTS.out.pose_file)
    GENERATE_MULTI_MOUSE_TRACKLETS(num_animals, PREDICT_MULTI_MOUSE_IDENTITY.out.pose_file)
    // Figure out how to copy the final pose output out
}
