include { GENERATE_GAIT_H5; GENERATE_GAIT_BIN } from "./../../nextflow/modules/gait"
include { MERGE_FEATURE_ROWS } from "./../../nextflow/modules/utils"


workflow SINGLE_MOUSE_V2_FEATURES {
    take:
    // tuple of video_file and pose_file from SINGLE_MOUSE_TRACKING
    input_pose_v2_batch
    
    main:
    // Gait features
    gait_h5_files = GENERATE_GAIT_H5(input_pose_v2_batch).gait_file
    gait_bins = [10, 15, 20, 25]
    binned_gait_results = GENERATE_GAIT_BIN(gait_h5_files, Channel.fromList(gait_bins)).gait_bin_csv.collect()
    gait_results = MERGE_FEATURE_ROWS(binned_gait_results, "gait", 1)

    // Morphometrics features
    

    emit:
    gait_results
}