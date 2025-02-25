

nextflow.enable.dsl=2

include { PREPARE_DATA } from './nextflow/workflows/io'

if (!params.workflow) {
    println "Missing workflow parameter"
    System.exit(1)
}
include { SINGLE_MOUSE_TRACKING } from './nextflow/workflows/single_mouse_pipeline'
include { SINGLE_MOUSE_V2_FEATURES; SINGLE_MOUSE_V6_FEATURES } from './nextflow/workflows/feature_generation'
include { MULTI_MOUSE_TRACKING } from './nextflow/workflows/multi_mouse_pipeline'
include { MANUALLY_CORRECT_CORNERS; INTEGRATE_CORNER_ANNOTATIONS } from './nextflow/workflows/sleap_manual_correction'
include { ADD_DUMMY_VIDEO } from './nextflow/modules/utils'

/*
 * Combine input_data and input_batch into a single list
 */
all_files = []
if (params.input_data != null){
    all_files.add(params.input_data)
}
if (params.input_batch != null){
    all_files.addAll(file(params.input_batch).text.readLines())
}

if (all_files.size() == 0){
    println "Missing any data to process, please assign either input_data or input_batch"
    System.exit(1)
}

/*
 * Run the selected workflow
 */
workflow{
    // Download the data locally if necessary
    PREPARE_DATA(Channel.fromList(all_files), params.location)

    // Generate pose files
    if (params.workflow == "single-mouse"){
        SINGLE_MOUSE_TRACKING(PREPARE_DATA.out.out_file)
        all_v2_outputs = SINGLE_MOUSE_TRACKING.out[0].collect()
        v6_with_corners = SINGLE_MOUSE_TRACKING.out[1].collect()
        v6_without_corners = SINGLE_MOUSE_TRACKING.out[2].collect()

        // Pose v2 features
        pose_v2_results = SINGLE_MOUSE_V2_FEATURES(all_v2_outputs)

        // Pose v6 features
        SINGLE_MOUSE_V6_FEATURES(v6_with_corners)

        // Manual corner correction
        manual_output = MANUALLY_CORRECT_CORNERS(v6_without_corners, params.corner_frame)
    }
    if (params.workflow == "corner-corrected-features"){
        // Integrate annotations back into pose files
        INTEGRATE_CORNER_ANNOTATIONS(Channel.fromList(all_files), params.sleap_file)
        ADD_DUMMY_VIDEO(INTEGRATE_SLEAP_CORNER_ANNOTATIONS.out)
        paired_video_and_pose = ADD_DUMMY_VIDEO.out[0].collect()

        // Pose v6 features
        SINGLE_MOUSE_V6_FEATURES(paired_video_and_pose)
    }
    if (params.workflow == "multi-mouse"){
        MULTI_MOUSE_TRACKING(PREPARE_DATA.out.video_file, params.num_mice)
    }
}

