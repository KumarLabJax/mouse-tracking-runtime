nextflow.enable.dsl=2

include { PREPARE_DATA } from './nextflow/workflows/io'
include { MULTI_MOUSE_TRACKING } from './nextflow/workflows/multi_mouse_pipeline'
include { COMPRESS_VIDEO_CRF; COMPRESS_VIDEO_BR; COMPUTE_VIDEO_DIFFERENCE } from './nextflow/workflows/compression'
include { RENDER_POSE } from './nextflow/workflows/pose_qc'

workflow {
    crf_set = channel.of(17, 23, 29)
    bitrate_set = channel.of(1000, 500, 100)
    keyframe_set = channel.of(3000)

    // Iterate over all combinations of CRF, bitrate, and keyframe interval
    in_videos = PREPARE_DATA(params.input_batch, params.location, false).file_processing_channel
    crf_videos = COMPRESS_VIDEO_CRF(in_videos.combine(crf_set).combine(keyframe_set)).files
    bitrate_videos = COMPRESS_VIDEO_BR(in_videos.combine(bitrate_set).combine(keyframe_set)).files
    all_videos = in_videos.concat(crf_videos).concat(bitrate_videos)

    pose_out = MULTI_MOUSE_TRACKING(all_videos, params.num_mice).pose_v6
    RENDER_POSE(pose_out)
}