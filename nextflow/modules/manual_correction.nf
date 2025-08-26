/**
 * This module contains process definitions related to manually correcting pose data using SLEAP.
 */

/**
 * Extracts a frame from a video as a png image.
 *
 * @param tuple
 *  - video The input video file.
 *  - pose_file The input pose file (not used in this process).
 * @param frame_index The index of the frame to extract.
 *
 * @return frame A png image of the extracted frame.
 */
process EXTRACT_VIDEO_FRAME {
    label "sleap"

    input:
    tuple path(video), path(pose_file)
    val frame_index

    output:
    path "${video.baseName}.png", emit: frame

    script:
    """
    ffmpeg -i ${video} -vf "select=gte(n\\,${frame_index}),setpts=PTS-STARTPTS" -vframes 1 ${video.getBaseName().replaceAll('%', '%%')}.png
    """
}

/**
 * Adds a set of frames to a new SLEAP project file for arena corner annotation.
 *
 * @param video_frames A list of png images to be added to the SLEAP project.
 *
 * @return sleap_file The generated SLEAP project file containing the frames and arena corner skeleton for annotation.
 */
process ADD_EXAMPLES_TO_SLEAP {
    label "sleap"

    input:
    path video_frames

    output:
    path "corner-correction.slp", emit: sleap_file

    script:
    """
    #!/usr/bin/env python3

    import sleap
    from sleap.io.video import Video
    from sleap.skeleton import Skeleton

    skeleton_obj = Skeleton("arena_corners")
    skeleton_obj.add_nodes(["corners_kp0", "corners_kp1", "corners_kp2", "corners_kp3"])
    skeleton_obj.add_edge("corners_kp0", "corners_kp1")
    skeleton_obj.add_edge("corners_kp1", "corners_kp2")
    skeleton_obj.add_edge("corners_kp2", "corners_kp3")

    labels_obj = sleap.Labels(skeletons=[skeleton_obj])
    video_frames = [${video_frames.collect { element -> "\"${element.toString()}\"" }.join(', ')}]
    for frame in video_frames:
        new_video = Video.from_filename(frame)
        labels_obj.add_video(new_video)
        labels_obj.add_suggestion(new_video, 0)
    
    sleap.Labels.save_file(labels_obj, "corner-correction.slp", all_labeled=True, save_frame_data=True, suggested=True)
    """
}

/**
 * Integrates SLEAP corner annotations into a pose file.
 *
 * @param pose_file The input pose file to be corrected.
 * @param sleap_file The SLEAP project file containing the corner annotations.
 *
 * @return pose_file The corrected pose file with integrated corner annotations.
 */
process INTEGRATE_SLEAP_CORNER_ANNOTATIONS {
    label "sleap_io"

    input:
    path pose_file
    path sleap_file

    output:
    path "${pose_file.baseName}_corrected.h5", emit: pose_file

    script:
    """
    cp ${pose_file} ${pose_file.baseName}_corrected.h5
    python /mouse-tracking-runtime/support_code/static-object-correct.py --pose-file ${pose_file.baseName}_corrected.h5 --sleap-annotations ${sleap_file}
    """
}