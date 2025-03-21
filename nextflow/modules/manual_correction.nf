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