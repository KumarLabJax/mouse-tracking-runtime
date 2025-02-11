

nextflow.enable.dsl=2

include { PREPARE_DATA } from './nextflow/workflows/io'

if (!params.workflow) {
    println "Missing workflow parameter"
    System.exit(1)
}
include { SINGLE_MOUSE_TRACKING } from './nextflow/workflows/single_mouse_pipeline'
include { SINGLE_MOUSE_V2_FEATURES; SINGLE_MOUSE_V6_FEATURES } from './nextflow/workflows/feature_generation'
include { MULTI_MOUSE_TRACKING } from './nextflow/workflows/multi_mouse_pipeline'
include { QC_SINGLE_MOUSE } from './nextflow/modules/single_mouse'
include { SELECT_COLUMNS;
          SUBSET_PATH_BY_VAR as WITH_CORNERS;
          SUBSET_PATH_BY_VAR as WITHOUT_CORNERS;
          PUBLISH_RESULT_FILE as PUBLISH_SM_TRIMMED_VID;
          PUBLISH_RESULT_FILE as PUBLISH_SM_POSE_V2;
          PUBLISH_RESULT_FILE as PUBLISH_SM_POSE_V6;
          PUBLISH_RESULT_FILE as PUBLISH_SM_V6_FEATURES;
          PUBLISH_RESULT_FILE as PUBLISH_FBOLI } from './nextflow/modules/utils'

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

        // Publish the pose results
        trimmed_video_files = all_v2_outputs.map { video, pose ->
            tuple(video, "${video.name}")
        }
        PUBLISH_SM_TRIMMED_VID(trimmed_video_files)
        v2_poses_renamed = all_v2_outputs.map { video, pose ->
            tuple(pose, "${video.baseName}_pose_est_v2.h5")
        }
        PUBLISH_SM_POSE_V2(v2_poses_renamed)
        v6_poses_renamed = all_v6_outputs.map { video, pose ->
            tuple(pose, "${video.baseName}_pose_est_v6.h5")
        }
        PUBLISH_SM_POSE_V6(v6_poses_renamed)

        // Pose v2 branch calculations
        SINGLE_MOUSE_V2_FEATURES(all_v2_outputs)

        // Only continue processing files that generate corners
        def split_criteria = multiMapCriteria { f, v ->
            present: v == "True" ? f : null
            missing: v == "False" ? f : null
        }
        joined_channel = SELECT_COLUMNS(QC_SINGLE_MOUSE.out, 'pose_file', 'corners_present')
            .splitCsv(header: true, sep: ',')
            .map(row -> [row.pose_file, row.corners_present])
        joined_channel.view() {v -> "All files: ${v}"}
        split_channel = joined_channel.multiMap(split_criteria)
        split_channel.present.view() {v -> "Present corners: ${v}"}
        split_channel.missing.view() {v -> "Missing corners: ${v}"}
        // v6_with_corners = WITH_CORNERS(all_v6_outputs, split_channel.present.flatten(), 'tmp')
        v6_with_corners = all_v6_outputs.filter { video, pose ->
            split_channel.present.flatten().toList().contains(pose) ? [video, pose] : null
        }
        SINGLE_MOUSE_V6_FEATURES(v6_with_corners)

        // Publish the feature results
        feature_file = SINGLE_MOUSE_V6_FEATURES.out[0].collect()
        fecal_boli = SINGLE_MOUSE_V6_FEATURES.out[1].collect()
        feature_outputs = feature_file.map { feature_file ->
            tuple(feature_file, "features.csv")
        }
        PUBLISH_SM_V6_FEATURES(feature_outputs)
        fecal_boli_outputs = fecal_boli.map { fecal_boli ->
            tuple(fecal_boli, "fecal_boli.csv")
        }
        PUBLISH_FBOLI(fecal_boli_outputs)

        // v6_without_corners = Channel.fromPath(SELECT_COLUMNS.out.without_corners)
        // ADD_TO_MANUAL_CORNER_CORRECTION(v6_without_corners)
    }
    if (params.workflow == "multi-mouse"){
        MULTI_MOUSE_TRACKING(PREPARE_DATA.out.video_file, params.num_mice)
    }

    // Move the final pose files to the appropriate location
    //POSTPROCESS_DATA(all_files, params.location)
}

