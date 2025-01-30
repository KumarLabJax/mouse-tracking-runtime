

nextflow.enable.dsl=2

include { PREPARE_DATA } from './nextflow/workflows/io'

if (!params.workflow) {
    println "Missing workflow parameter"
    System.exit(1)
}
include { SINGLE_MOUSE_TRACKING } from './nextflow/workflows/single_mouse_pipeline'
include { SINGLE_MOUSE_V2_FEATURES } from './nextflow/workflows/feature_generation'
include { MULTI_MOUSE_TRACKING } from './nextflow/workflows/multi_mouse_pipeline'
include { QC_SINGLE_MOUSE } from './nextflow/modules/single_mouse'
include { FILTER_QC_WITH_CORNERS } from './nextflow/modules/utils'

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
        SINGLE_MOUSE_TRACKING(PREPARE_DATA.out.video_file)
        // [0] contains v2 pose files and [1] contains v6 pose files
        all_v2_outputs = SINGLE_MOUSE_TRACKING.out[0].collect()
        all_v6_outputs = SINGLE_MOUSE_TRACKING.out[1].collect()
        QC_SINGLE_MOUSE(all_v6_outputs, params.clip_duration, params.batch_name)

        // Pose v2 branch calculations
        SINGLE_MOUSE_V2_FEATURES(all_v2_outputs)

        // Only continue processing files that generate corners
        // FILTER_QC_WITH_CORNERS(QC_SINGLE_MOUSE.out)
        // v6_with_corners = Channel.fromPath(FILTER_QC_WITH_CORNERS.out.with_corners)
        // SINGLE_MOUSE_V6_FEATURES(v6_with_corners)
        // v6_without_corners = Channel.fromPath(FILTER_QC_WITH_CORNERS.out.without_corners)
        // ADD_TO_MANUAL_CORNER_CORRECTION(v6_without_corners)
    }
    if (params.workflow == "multi-mouse"){
        MULTI_MOUSE_TRACKING(PREPARE_DATA.out.video_file, params.num_mice)
    }

    // Move the final pose files to the appropriate location
    //POSTPROCESS_DATA(all_files, params.location)
}

