include { EXTRACT_VIDEO_FRAME; ADD_EXAMPLES_TO_SLEAP; INTEGRATE_SLEAP_CORNER_ANNOTATIONS } from "${projectDir}/nextflow/modules/manual_correction"
include { PUBLISH_RESULT_FILE as PUBLISH_SM_MANUAL_CORRECT } from "${projectDir}/nextflow/modules/utils"

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

workflow INTEGRATE_CORNER_ANNOTATIONS {
    take:
    pose_files
    sleap_file

    main:
    corrected_poses = INTEGRATE_SLEAP_CORNER_ANNOTATIONS(pose_files, sleap_file).pose_file

    emit:
    corrected_poses
}
