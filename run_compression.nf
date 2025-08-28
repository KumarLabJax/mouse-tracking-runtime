nextflow.enable.dsl=2

include { PREPARE_DATA } from './nextflow/workflows/io'
include { MULTI_MOUSE_TRACKING } from './nextflow/workflows/multi_mouse_pipeline'

/**
 * Video compression using bitrate control
 *
 * @param tuple
 *  - video_file The input video file to compress
 *  - bitrate The target bitrate for the compression
 *  - keyframe_interval The keyframe interval for the compression
 *
 * @return tuple files
 *  - Original video
 *  - Compressed video
 */
process COMPRESS_VIDEO_CRF {
    label "compression"
    publishDir "compressed/compressed/", mode:'copy'

    input:
    tuple path(video_file), val(crf), val(keyframe_interval)

    output:
    tuple path(video_file), path("${video_file.baseName}_g${keyframe_interval}_crf${crf}.mp4"), emit: files

    script:
    """
    ffmpeg -i ${video_file} -c:v libx264 -pix_fmt yuv420p -preset slow -crf ${crf} -g ${keyframe_interval} -f mp4 ${video_file.baseName}_g${keyframe_interval}_crf${crf}.mp4
    """
}

/**
 * Video compression using bitrate control
 *
 * @param tuple
 *  - video_file The input video file to compress
 *  - bitrate The target bitrate for the compression
 *  - keyframe_interval The keyframe interval for the compression
 *
 * @return tuple files
 *  - Original video
 *  - Compressed video
 */
process COMPRESS_VIDEO_BR {
    label "compression"
    publishDir "compressed/compressed/", mode:'copy'

    input:
    tuple path(video_file), val(bitrate), val(keyframe_interval)

    output:
    tuple path(video_file), path("${video_file.baseName}_r${bitrate}_g${keyframe_interval}.mp4"), emit: files

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
 */
process COMPUTE_VIDEO_DIFFERENCE {
    label "tracking"
    publishDir "compressed/difference/", mode:'copy'

    input:
    tuple path(in_video1), path(in_video2)

    output:
    path "${in_video2.baseName}_diff.mp4"

    script:
    """
    ffmpeg -i ${in_video1} -i ${in_video2} -filter_complex '[0:v]setsar=1:1[v1];[v1][1:v]blend=all_mode=difference128' -c:v mpeg4 -q 0 ${in_video2.baseName}_diff.mp4
    """
}

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

workflow {
    crf_set = channel.of(17, 23, 29)
    bitrate_set = channel.of(1000, 500, 100)
    keyframe_set = channel.of(3000)

    // Iterate over all combinations of CRF, bitrate, and keyframe interval
    in_videos = PREPARE_DATA(params.input_batch, params.location, false).file_processing_channel
    crf_videos = COMPRESS_VIDEO_CRF(in_videos.combine(crf_set).combine(keyframe_set)).files
    bitrate_videos = COMPRESS_VIDEO_BR(in_videos.combine(bitrate_set).combine(keyframe_set)).files
    all_videos = in_videos.concat(crf_videos.map { original, compressed -> compressed }).concat(bitrate_videos.map { original, compressed -> compressed })

    COMPUTE_VIDEO_DIFFERENCE(crf_videos.concat(bitrate_videos))
    pose_out = MULTI_MOUSE_TRACKING(all_videos, params.num_mice).pose_v6
    RENDER_POSE(pose_out)
}