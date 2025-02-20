

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
include { MANUALLY_CORRECT_CORNERS } from './nextflow/workflows/sleap_manual_correction'
include { SELECT_COLUMNS;
          SUBSET_PATH_BY_VAR as WITH_CORNERS;
          SUBSET_PATH_BY_VAR as WITHOUT_CORNERS;
          PUBLISH_RESULT_FILE as PUBLISH_SM_TRIMMED_VID;
          PUBLISH_RESULT_FILE as PUBLISH_SM_POSE_V2;
          PUBLISH_RESULT_FILE as PUBLISH_GAIT;
          PUBLISH_RESULT_FILE as PUBLISH_MORPHOMETRICS;
          PUBLISH_RESULT_FILE as PUBLISH_SM_POSE_V6;
          PUBLISH_RESULT_FILE as PUBLISH_SM_POSE_V6_NOCORN;
          PUBLISH_RESULT_FILE as PUBLISH_SM_QC;
          PUBLISH_RESULT_FILE as PUBLISH_SM_V6_FEATURES;
          PUBLISH_RESULT_FILE as PUBLISH_FBOLI;
          PUBLISH_RESULT_FILE as PUBLISH_SM_MANUAL_CORRECT;
          REMOVE_URLIFY_FIELDS as NOURL_QC;
          REMOVE_URLIFY_FIELDS as NOURL_JABS;
          REMOVE_URLIFY_FIELDS as NOURL_GAIT;
          REMOVE_URLIFY_FIELDS as NOURL_MORPH;
          REMOVE_URLIFY_FIELDS as NOURL_FBOLI;
          DELETE_ROW as DELETE_DEFAULT_JABS;
          DELETE_ROW as DELETE_DEFAULT_FBOLI } from './nextflow/modules/utils'

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
        // [0] contains v2 pose files and [1] contains v6 pose files
        all_v2_outputs = SINGLE_MOUSE_TRACKING.out[0].collect()
        all_v6_outputs = SINGLE_MOUSE_TRACKING.out[1].collect()
        QC_SINGLE_MOUSE(all_v6_outputs, params.clip_duration, params.batch_name)
        qc_output = QC_SINGLE_MOUSE.out.qc_file
        PUBLISH_SM_QC(NOURL_QC(qc_output).map { file -> tuple(file, "qc_${params.batch_name}.csv") })

        // Publish the pose results
        trimmed_video_files = all_v2_outputs.map { video, pose ->
            tuple(video, "results/${video.name.replace("%20", "/")}")
        }
        PUBLISH_SM_TRIMMED_VID(trimmed_video_files)
        v2_poses_renamed = all_v2_outputs.map { video, pose ->
            tuple(pose, "results/${video.baseName.replace("%20", "/")}_pose_est_v2.h5")
        }
        PUBLISH_SM_POSE_V2(v2_poses_renamed)

        // Pose v2 branch calculations
        pose_v2_results = SINGLE_MOUSE_V2_FEATURES(all_v2_outputs)
        gait_outputs = NOURL_GAIT(pose_v2_results[0]).map { feature_file ->
            tuple(feature_file, "gait.csv")
        }
        PUBLISH_GAIT(gait_outputs)
        morphometric_outputs = NOURL_MORPH(pose_v2_results[1]).map { feature_file ->
            tuple(feature_file, "morphometrics.csv")
        }
        PUBLISH_MORPHOMETRICS(morphometric_outputs)

        // Only continue processing files that generate corners
        joined_channel = SELECT_COLUMNS(QC_SINGLE_MOUSE.out, 'pose_file', 'corners_present')
            .splitCsv(header: true, sep: ',')
            .map(row -> [row.pose_file, row.corners_present])
        // Split qc filenames into present and missing
        split_channel = joined_channel.branch { v, c ->
            present: c.contains("True")
                return v
            missing: c.contains("False")
                return v
        }
        // Split path channel with defaults
        branched = all_v6_outputs.branch { video, pose ->
            present: split_channel.present.ifEmpty("INVALID_POSE_FILE").toList().contains(pose.toString())
            missing: split_channel.missing.ifEmpty("INVALID_POSE_FILE").toList().contains(pose.toString())
        }
        // v6_with_corners = branched.present.ifEmpty([Path(params.default_feature_input[0]), Path(params.default_feature_input[1])])
        // v6_without_corners = branched.missing.ifEmpty([Path(params.default_manual_correction_input[0]), Path(params.default_manual_correction_input[1])])
        v6_with_corners = branched.present.ifEmpty(params.default_feature_input)
        v6_without_corners = branched.missing.ifEmpty(params.default_manual_correction_input)
        SINGLE_MOUSE_V6_FEATURES(v6_with_corners)

        // Publish the feature results
        feature_file = SINGLE_MOUSE_V6_FEATURES.out[0].collect()
        fecal_boli = SINGLE_MOUSE_V6_FEATURES.out[1].collect()
        feature_outputs = NOURL_JABS(DELETE_DEFAULT_JABS(feature_file, "${file(params.default_feature_input[0]).baseName}")).map { feature_file ->
            tuple(feature_file, "features.csv")
        }
        PUBLISH_SM_V6_FEATURES(feature_outputs)
        fecal_boli_outputs = NOURL_FBOLI(DELETE_DEFAULT_FBOLI(fecal_boli, "${file(params.default_feature_input[0]).baseName}")).map { fecal_boli ->
            tuple(fecal_boli, "fecal_boli.csv")
        }
        PUBLISH_FBOLI(fecal_boli_outputs)

        v6_poses_renamed = v6_with_corners.map { video, pose ->
            tuple(pose, "results/${video.baseName.replace("%20", "/")}_pose_est_v6.h5")
        }
        PUBLISH_SM_POSE_V6(v6_poses_renamed)
        // Corners that failed are placed in a separate folder with url-ified names
        v6_no_corners_renamed = v6_without_corners.map { video, pose ->
            tuple(pose, "failed_corners/${file(video).baseName}_pose_est_v6.h5")
        }
        PUBLISH_SM_POSE_V6_NOCORN(v6_no_corners_renamed)

        v6_with_corners.view() { println "WITH CORNERS: $it" }
        v6_without_corners.view() { println "WITHOUT CORNERS: $it" }
        manual_output = MANUALLY_CORRECT_CORNERS(v6_without_corners, params.corner_frame)
        manual_correction_output = manual_output.map { sleap_file ->
            tuple(sleap_file, "manual_corner_correction.slp")
        }
        PUBLISH_SM_MANUAL_CORRECT(manual_correction_output)
        println "Workflow commit ID: ${workflow.commitId ?: 'N/A'}"
    }
    if (params.workflow == "multi-mouse"){
        MULTI_MOUSE_TRACKING(PREPARE_DATA.out.video_file, params.num_mice)
    }
}

