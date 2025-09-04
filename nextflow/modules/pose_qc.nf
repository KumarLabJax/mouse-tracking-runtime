/**
 * Render pose on video
 *
 * @param in_video The input video file
 * @param in_pose The input pose file
 *
 * @return Rendered video
 */
process RENDER_POSE {
    label "tracking"
    publishDir "compressed/pose/", mode:'copy'

    input:
    tuple path(in_video), path(in_pose)

    output:
    path "${in_video.baseName}_pose.mp4"

    script:
    """
    python3 /kumar_lab_models/mouse-tracking-runtime/render_pose.py --in-vid ${in_video} --in-pose ${in_pose} --out-vid ${in_video.baseName}_pose.mp4
    """
}
