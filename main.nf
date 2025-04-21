

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
include { ADD_DUMMY_VIDEO, validateInputFile } from './nextflow/modules/utils'

/*
 * Convert input_batch into a single list
 */
all_files = []
invalid_files = []
valid_files = []

if (params.input_batch != null) {
    def batch_lines = file(params.input_batch).text.readLines()
    
    // Validate each file in the batch
    batch_lines.each { file_path ->
        def (is_valid, error_message) = validateInputFile(file_path)
        
        if (is_valid) {
            valid_files.add(file_path)
        } else {
            invalid_files.add([file_path, error_message])
        }
    }
    
    // Report any invalid files
    if (invalid_files.size() > 0) {
        println "The following files failed validation:"
        invalid_files.each { file_path, error_message ->
            println "  - ${error_message}"
        }
        
        // If all files are invalid, exit
        if (valid_files.size() == 0) {
            println "No valid files to process. Exiting."
            System.exit(1)
        }
        
        // Otherwise, continue with valid files and warn the user
        println "Continuing with ${valid_files.size()} valid files out of ${batch_lines.size()} total files."
    }
    
    all_files.addAll(valid_files)
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
        // Use a channel of `all_files` instead of `PREPARE_DATA.out.out_file`
        INTEGRATE_CORNER_ANNOTATIONS(Channel.fromList(all_files), params.sleap_file)
        ADD_DUMMY_VIDEO(INTEGRATE_CORNER_ANNOTATIONS.out, params.clip_duration)
        paired_video_and_pose = ADD_DUMMY_VIDEO.out[0]

        // Pose v6 features
        SINGLE_MOUSE_V6_FEATURES(paired_video_and_pose)
    }
    if (params.workflow == "single-mouse-v6-features"){
        // Generate features from pose_v6 files
        ADD_DUMMY_VIDEO(PREPARE_DATA.out.out_file, params.clip_duration)
        paired_video_and_pose = ADD_DUMMY_VIDEO.out[0]
        SINGLE_MOUSE_V6_FEATURES(paired_video_and_pose)
    }
    // if (params.workflow == "multi-mouse"){
    //     MULTI_MOUSE_TRACKING(PREPARE_DATA.out.video_file, params.num_mice)
    // }
}

