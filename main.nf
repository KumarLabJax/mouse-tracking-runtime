nextflow.enable.dsl=2

include { PREPARE_DATA } from './nextflow/workflows/io'

if (!params.workflow) {
    println "Missing workflow parameter"
    System.exit(1)
}
include { SINGLE_MOUSE_TRACKING; SPLIT_BY_CORNERS } from './nextflow/workflows/single_mouse_pipeline'
include { SINGLE_MOUSE_V2_FEATURES; SINGLE_MOUSE_V6_FEATURES } from './nextflow/workflows/feature_generation'
include { MULTI_MOUSE_TRACKING } from './nextflow/workflows/multi_mouse_pipeline'
include { MANUALLY_CORRECT_CORNERS; INTEGRATE_CORNER_ANNOTATIONS } from './nextflow/workflows/sleap_manual_correction'
include { ADD_DUMMY_VIDEO } from './nextflow/modules/utils'

/*
 * Run the selected workflow
 */
workflow{
    // Generate pose files
    if (params.workflow == "single-mouse"){
        PREPARE_DATA(params.input_batch, params.location, false)
        SINGLE_MOUSE_TRACKING(PREPARE_DATA.out.file_processing_channel)
        v2_outputs = SINGLE_MOUSE_TRACKING.out[0]
        all_v6_outputs = SINGLE_MOUSE_TRACKING.out[1]
        // Split and publish pose_v6 files depending on if corners were successful
        SPLIT_BY_CORNERS(all_v6_outputs)
        v6_with_corners = SPLIT_BY_CORNERS.out[0]
        v6_without_corners = SPLIT_BY_CORNERS.out[1]

        // Pose v2 features
        pose_v2_results = SINGLE_MOUSE_V2_FEATURES(v2_outputs)

        // Pose v6 features
        SINGLE_MOUSE_V6_FEATURES(v6_with_corners)

        // Manual corner correction
        manual_output = MANUALLY_CORRECT_CORNERS(v6_without_corners, params.corner_frame)
    }
    if (params.workflow == "single-mouse-corrected-corners"){
        // Integrate annotations back into pose files
        // This branch requires files to be local and already url-ified
        PREPARE_DATA(params.input_batch, params.location, true)
        INTEGRATE_CORNER_ANNOTATIONS(PREPATE_DATA.out.file_processing_channel, params.sleap_file)
        ADD_DUMMY_VIDEO(INTEGRATE_CORNER_ANNOTATIONS.out, params.clip_duration)
        paired_video_and_pose = ADD_DUMMY_VIDEO.out[0]

        // Pose v6 features
        SINGLE_MOUSE_V6_FEATURES(paired_video_and_pose)
    }
    if (params.workflow == "single-mouse-v6-features"){
        PREPARE_DATA(params.input_batch, params.location, false)
        // Generate features from pose_v6 files
        ADD_DUMMY_VIDEO(PREPARE_DATA.out.out_file, params.clip_duration)
        paired_video_and_pose = ADD_DUMMY_VIDEO.out[0]
        SINGLE_MOUSE_V6_FEATURES(paired_video_and_pose)
    }
    if (params.workflow == "multi-mouse"){
        PREPARE_DATA(params.input_batch, params.location, false)
        MULTI_MOUSE_TRACKING(PREPARE_DATA.out.file_processing_channel, params.num_mice)
    }
}

