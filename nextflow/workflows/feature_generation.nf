include { GENERATE_GAIT_H5; GENERATE_GAIT_BIN } from "./../../nextflow/modules/gait"
include { GENERATE_FLEXIBILITY_INDEX; GENERATE_REAR_PAW_WIDTH } from "./../../nextflow/modules/flexibility"
include { MERGE_FEATURE_ROWS as MERGE_GAIT;
          MERGE_FEATURE_ROWS as MERGE_ANGLES;
          MERGE_FEATURE_ROWS as MERGE_DIST_AC;
          MERGE_FEATURE_ROWS as MERGE_DIST_B;
          MERGE_FEATURE_ROWS as MERGE_REAR_PAW_WIDTHS;
          MERGE_FEATURE_COLS } from "./../../nextflow/modules/utils"


workflow SINGLE_MOUSE_V2_FEATURES {
    take:
    // tuple of video_file and pose_file from SINGLE_MOUSE_TRACKING
    input_pose_v2_batch
    
    main:
    // Gait features
    gait_h5_files = GENERATE_GAIT_H5(input_pose_v2_batch).gait_file
    gait_bins = [10, 15, 20, 25]
    binned_gait_results = GENERATE_GAIT_BIN(gait_h5_files, Channel.fromList(gait_bins)).gait_bin_csv.collect()
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

    emit:
    gait_results
}