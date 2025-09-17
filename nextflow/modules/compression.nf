/**
 * Video compression using quality control
 *
 * @param tuple
 *  - video_file The input video file to compress
 *  - crf The constant rate factor for the compression
 *  - keyframe_interval The keyframe interval for the compression
 *
 * @return file Path to compressed video
 */
process COMPRESS_VIDEO_CRF {
    label "cpu"
    // Any environment with ffmpeg installed should work here.
    label "tracking"
    label "r_compression"

    input:
    tuple path(video_file), val(crf), val(keyframe_interval)

    output:
    path("${video_file.baseName}_g${keyframe_interval}_crf${crf}.mp4"), emit: file

    script:
    """
    ffmpeg -i ${video_file} -c:v libx264 -pix_fmt yuv420p -preset veryfast -crf ${crf} -g ${keyframe_interval} -f mp4 ${video_file.baseName}_g${keyframe_interval}_crf${crf}.mp4
    """
}

/**
 * Video compression using bitrate control
 *
 * Runs a 2-pass encoding of the video with a target bitrate.
 *
 * @param tuple
 *  - video_file The input video file to compress
 *  - bitrate The target bitrate for the compression
 *  - keyframe_interval The keyframe interval for the compression
 *
 * @return file Path to compressed video
 *
 * @note COMPRESS_VIDEO_CRF will typically produce better videos with a low signal-to-noise ratio.
 */
process COMPRESS_VIDEO_BR {
    label "cpu"
    // Any environment with ffmpeg installed should work here.
    label "tracking"
    label "r_compression"

    input:
    tuple path(video_file), val(bitrate), val(keyframe_interval)

    output:
    path("${video_file.baseName}_r${bitrate}_g${keyframe_interval}.mp4"), emit: file

    script:
    """
    ffmpeg -i ${video_file} -c:v libx264 -b:v ${bitrate}k -maxrate ${bitrate}k -bufsize \$((${bitrate}*2))k -g ${keyframe_interval} -pass 1 -f mp4 ${video_file.baseName}_r${bitrate}_g${keyframe_interval}_PASS1.mp4 && \
    ffmpeg -i ${video_file.baseName}_r${bitrate}_g${keyframe_interval}_PASS1.mp4 -c:v libx264 -b:v ${bitrate}k -maxrate ${bitrate}k -bufsize \$((${bitrate}*2))k -g ${keyframe_interval} -pass 2 ${video_file.baseName}_r${bitrate}_g${keyframe_interval}.mp4 && \
    rm ${video_file.baseName}_r${bitrate}_g${keyframe_interval}_PASS1.mp4 ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree

    """
}

/**
 * Computes a difference between two videos
 *
 * @param in_video1 The first input video file
 * @param in_video2 The second input video file
 *
 * @return The output video file with the computed difference
 *
 * @note Difference videos are useful in comparing changes due to compression.
 */
process COMPUTE_VIDEO_DIFFERENCE {
    label "cpu"
    // Any environment with ffmpeg installed should work here.
    label "tracking"

    input:
    tuple path(in_video1), path(in_video2)

    output:
    path "${in_video2.baseName}_diff.mp4"

    script:
    """
    ffmpeg -i ${in_video1} -i ${in_video2} -filter_complex '[0:v]setsar=1:1[v1];[v1][1:v]blend=all_mode=difference128' -c:v mpeg4 -q 0 ${in_video2.baseName}_diff.mp4
    """
}
