/**
 * Render pose on video
 *
 * @param in_video The input video file
 * @param in_pose The input pose file
 *
 * @return Rendered video
 *
 * @publish ./qc Rendered pose video
 */
process RENDER_POSE {
    label "tracking"
    publishDir "${params.pubdir}/qc", mode:'copy'

    input:
    tuple path(in_video), path(in_pose)

    output:
    path "${in_video.baseName}_pose.mp4"

    script:
    """
    mouse-tracking utils render-pose ${in_video} ${in_pose} ${in_video.baseName}_pose.mp4
    """
}

/**
 * Render fecal boli on a frame
 *
 * @param in_video The input video file
 * @param in_pose The input pose file
 *
 * @return Rendered fecal boli video
 *
 * @publish ./qc Rendered fecal boli video
 */
process RENDER_BOLI {
    label "tracking"
    publishDir "${params.pubdir}/qc", mode:'copy'

    input:
    tuple path(in_video), path(in_pose)

    output:
    path "${in_video.baseName}_boli.avi"

    script:
    """
    mouse-tracking utils render-fecal-boli ${in_video} ${in_pose} ${in_video.baseName}_boli.avi
    """
}
