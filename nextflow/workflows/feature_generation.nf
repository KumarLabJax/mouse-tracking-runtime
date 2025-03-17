include { GENERATE_GAIT_H5; GENERATE_GAIT_BIN } from "${projectDir}/nextflow/modules/gait"
include { GENERATE_FLEXIBILITY_INDEX; GENERATE_REAR_PAW_WIDTH } from "${projectDir}/nextflow/modules/flexibility"
include { GET_WORKFLOW_VERSION;
          MERGE_FEATURE_ROWS as MERGE_GAIT;
          MERGE_FEATURE_ROWS as MERGE_ANGLES;
          MERGE_FEATURE_ROWS as MERGE_DIST_AC;
          MERGE_FEATURE_ROWS as MERGE_DIST_B;
          MERGE_FEATURE_ROWS as MERGE_REAR_PAW_WIDTHS;
          MERGE_FEATURE_ROWS as MERGE_FECAL_BOLI;
          MERGE_FEATURE_ROWS as MERGE_JABS;
          MERGE_FEATURE_COLS;
          DELETE_ROW as DELETE_DEFAULT_JABS;
          DELETE_ROW as DELETE_DEFAULT_FBOLI;
          PUBLISH_RESULT_FILE as PUBLISH_SM_V6_FEATURES;
          PUBLISH_RESULT_FILE as PUBLISH_FBOLI;
          PUBLISH_RESULT_FILE as PUBLISH_GAIT;
          PUBLISH_RESULT_FILE as PUBLISH_MORPHOMETRICS;
          REMOVE_URLIFY_FIELDS as NOURL_JABS;
          REMOVE_URLIFY_FIELDS as NOURL_FBOLI;
          REMOVE_URLIFY_FIELDS as NOURL_GAIT;
          REMOVE_URLIFY_FIELDS as NOURL_MORPH;
          ADD_COLUMN as ADD_VERSION_GAIT;
          ADD_COLUMN as ADD_VERSION_MORPH;
          ADD_COLUMN as ADD_VERSION_JABS;
          ADD_COLUMN as ADD_VERSION_FBOLI;
          FEATURE_TO_LONG;
          LONG_TO_WIDE; } from "${projectDir}/nextflow/modules/utils"
include { GENERATE_FEATURE_CACHE;
          PREDICT_CLASSIFIERS;
          GENERATE_BEHAVIOR_TABLES;
          PREDICT_HEURISTICS;
          BEHAVIOR_TABLE_TO_FEATURES } from "${projectDir}/nextflow/modules/jabs_classifiers"
include { EXTRACT_FECAL_BOLI_BINS } from "${projectDir}/nextflow/modules/fecal_boli"

workflow SINGLE_MOUSE_V2_FEATURES {
    take:
    // tuple of video_file and pose_file from SINGLE_MOUSE_TRACKING
    input_pose_v2_batch
    
    main:
    // Gait features
    gait_h5_files = GENERATE_GAIT_H5(input_pose_v2_batch).gait_file
    gait_bins = [10, 15, 20, 25]
    gait_combinations = gait_h5_files.combine(Channel.fromList(gait_bins))
    binned_gait_results = GENERATE_GAIT_BIN(gait_combinations).gait_bin_csv.collect()
    gait_results = MERGE_GAIT(binned_gait_results, "gait", 1)

    // Morphometrics features
    flexibility = GENERATE_FLEXIBILITY_INDEX(input_pose_v2_batch)
    combined_angle_results = MERGE_ANGLES(flexibility.angles.collect(), "flexibility_angles", 1).merged_features
    combined_ac_results = MERGE_DIST_AC(flexibility.dist_ac.collect(), "flexibility_dist_ac", 1).merged_features
    combined_b_results = MERGE_DIST_B(flexibility.dist_b.collect(), "flexibility_dist_b", 1).merged_features

    rear_paws = GENERATE_REAR_PAW_WIDTH(input_pose_v2_batch)
    combined_rearpaw_results = MERGE_REAR_PAW_WIDTHS(rear_paws.rearpaw.collect(), "rearpaw", 1).merged_features

    all_morphometrics = combined_angle_results.concat(combined_ac_results, combined_b_results, combined_rearpaw_results).collect()

    morphometrics_results = MERGE_FEATURE_COLS(all_morphometrics, "NetworkFilename", "morphometrics")

    // Publish the results
    workflow_version = GET_WORKFLOW_VERSION().version
    gait_outputs = NOURL_GAIT(ADD_VERSION_GAIT(gait_results, "nextflow_version", workflow_version)).map { feature_file ->
        tuple(feature_file, "gait.csv")
    }
    PUBLISH_GAIT(gait_outputs)
    morphometric_outputs = NOURL_MORPH(ADD_VERSION_MORPH(morphometrics_results, "nextflow_version", workflow_version)).map { feature_file ->
        tuple(feature_file, "morphometrics.csv")
    }
    PUBLISH_MORPHOMETRICS(morphometric_outputs)

    emit:
    gait_results
    morphometrics_results
}

workflow SINGLE_MOUSE_V6_FEATURES {
    take:
    // tuple of video_file and pose_file from SINGLE_MOUSE_TRACKING
    input_pose_v6_batch

    main:
    cached_features = GENERATE_FEATURE_CACHE(input_pose_v6_batch).files
    // JABS Heuristic Classifiers
    heuristic_classifiers = params.heuristic_classifiers.collect { params.heuristic_classifier_folder + it + ".yaml" }
    heuristic_tables = PREDICT_HEURISTICS(cached_features, heuristic_classifiers)

    // JABS Behavior Classifiers
    // We let the inner prediction loop over classifiers because they write to a single file
    classifier_predictions = PREDICT_CLASSIFIERS(cached_features, params.single_mouse_classifiers)
    classifier_tables = GENERATE_BEHAVIOR_TABLES(classifier_predictions, params.single_mouse_classifiers)

    // Combine table data into feature file
    all_summary_tables = heuristic_tables
        .concat(classifier_tables)
        .map { bout_table, summary_table -> summary_table }
        .flatten()
        .combine(params.feature_bins)
    individual_behavior_features = BEHAVIOR_TABLE_TO_FEATURES(all_summary_tables)
    // Features are named columns (wide) split across multiple files
    // Transform them into long format so that we can row-concat without sorting
    long_feature_data = FEATURE_TO_LONG(individual_behavior_features, "MouseID")
    combined_long_features = MERGE_JABS(long_feature_data.collect(), "jabs_features", 1)
    wide_jabs_features = LONG_TO_WIDE(combined_long_features, "MouseID", "feature_name", "value")

    // Fecal Boli Extraction
    individual_fecal_boli = EXTRACT_FECAL_BOLI_BINS(input_pose_v6_batch)
    fecal_boli_table = MERGE_FECAL_BOLI(individual_fecal_boli.fecal_boli.collect(), "fecal_boli", 1)

    // Publish results
    workflow_version = GET_WORKFLOW_VERSION().version
    feature_outputs = NOURL_JABS(ADD_VERSION_JABS(DELETE_DEFAULT_JABS(wide_jabs_features, "${file(params.default_feature_input[0]).baseName}"), "nextflow_version", workflow_version)).map { feature_file ->
        tuple(feature_file, "features.csv")
    }
    PUBLISH_SM_V6_FEATURES(feature_outputs)
    fecal_boli_outputs = NOURL_FBOLI(ADD_VERSION_FBOLI(DELETE_DEFAULT_FBOLI(fecal_boli_table, "${file(params.default_feature_input[0]).baseName}"), "nextflow_version", workflow_version)).map { fecal_boli ->
        tuple(fecal_boli, "fecal_boli.csv")
    }
    PUBLISH_FBOLI(fecal_boli_outputs)

    emit:
    wide_jabs_features
    fecal_boli_table
}