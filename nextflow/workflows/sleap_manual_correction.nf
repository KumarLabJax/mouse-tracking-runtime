/**
 * This module contains workflows for manual correction of arena corner annotations.
 */
include { EXTRACT_VIDEO_FRAME; ADD_EXAMPLES_TO_SLEAP; INTEGRATE_SLEAP_CORNER_ANNOTATIONS } from "${projectDir}/nextflow/modules/manual_correction"
include { PUBLISH_RESULT_FILE as PUBLISH_SM_MANUAL_CORRECT } from "${projectDir}/nextflow/modules/utils"

/**
 * Workflow to generate a SLEAP file for manual correction of arena corners.
 *
 * @param path input_files Channel of input video files to extract frames from.
 * @param val frame_index The index of the frame to extract from each video.
 *
 * @return path sleap_file The SLEAP file containing the extracted frames for manual correction.
 *
 * @publish ./ Manual corner correction SLEAP file
 */
workflow MANUALLY_CORRECT_CORNERS {
    take:
    input_files
    frame_index

    main:
    video_frames = EXTRACT_VIDEO_FRAME(input_files, frame_index).frame
    sleap_file = ADD_EXAMPLES_TO_SLEAP(video_frames.collect()).sleap_file
    manual_correction_output = sleap_file.map { sleap_filename ->
        tuple(sleap_filename, "manual_corner_correction.slp")
    }
    PUBLISH_SM_MANUAL_CORRECT(manual_correction_output)

    emit:
    sleap_file
}

/**
 * Workflow to integrate manually corrected corner annotations from a SLEAP file back into pose files.
 *
 * @param path pose_files Channel of pose files to update with corrected corner annotations.
 * @param path sleap_file The SLEAP file containing the manually corrected corner annotations.
 *
 * @return path corrected_poses The channel of pose files updated with the corrected corner annotations.
 */
workflow INTEGRATE_CORNER_ANNOTATIONS {
    take:
    pose_files
    sleap_file

    main:
    corrected_poses = INTEGRATE_SLEAP_CORNER_ANNOTATIONS(pose_files, sleap_file).pose_file

    emit:
    corrected_poses
}
