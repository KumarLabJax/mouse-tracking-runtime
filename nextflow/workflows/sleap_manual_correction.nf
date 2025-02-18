include { EXTRACT_VIDEO_FRAME; ADD_EXAMPLES_TO_SLEAP; INTEGRATE_SLEAP_CORNER_ANNOTATIONS } from "./../../nextflow/modules/manual_correction"

workflow MANUALLY_CORRECT_CORNERS {
    take:
    input_files
    frame_index

    main:
    video_frames = EXTRACT_VIDEO_FRAME(input_files, frame_index).frame
    sleap_file = ADD_EXAMPLES_TO_SLEAP(video_frames.collect()).sleap_file

    emit:
    sleap_file
}

workflow INTEGRATE_CORNER_ANNOTATIONS {
    take:
    pose_files
    sleap_file

    main:
    corrected_poses = INTEGRATE_SLEAP_CORNER_ANNOTATIONS(pose_files, sleap_files).pose_file

    emit:
    corrected_poses
}
