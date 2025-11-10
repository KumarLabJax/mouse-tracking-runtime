nextflow.enable.dsl=2

/**
 * Retrieves files from Dropbox using rclone.
 *
 * @param files_to_transfer A file containing a list of files to transfer
 * @param rclone_prefix The rclone remote prefix where files are stored
 * @param rclone_config The rclone config file that provides remote dropbox authentication
 *
 * @return remote_files A file containing a list of the retrieved files with full paths.
 */
process GET_DATA_FROM_DROPBOX {
    container "/projects/kumar-lab/meta/images/mouse-tracking-runtime/rclone/latest.sif"
    cpus 1
    time 20.m
    memory 4.GB
    queue 'xfer'
    clusterOptions '-q xfer'
    maxForks 2
    errorStrategy 'ignore'
    
    input:
    path files_to_transfer
    val rclone_prefix
    path rclone_config

    output:
    path("fetched_files.txt"), emit: remote_files

    script:
    """
    rclone copy --ignore-checksum --config=${rclone_config} --transfers=1 --include-from ${files_to_transfer} ${rclone_prefix} retrieved_files/.
    find \$(pwd)/retrieved_files/ -type f > fetched_files.txt
    """
}

/**
 * Video compression
 *
 * @param tuple
 *  - filename The original input filename that be being compressed (to pass forward)
 *  - video_file The input video file to compress
 *
 * @return tuple files
 *  - filename Val Copy of original input filename compressed
 *  - file Path to compressed video
 */
process COMPRESS_VIDEO_CRF {
    container "/projects/kumar-lab/meta/images/mouse-tracking-runtime/runtime/latest.sif"
    cpus 2
    memory 1.GB
    time 2.hours
    array 200
    errorStrategy 'ignore'

    input:
    tuple val(filename), path(video_file)

    output:
    tuple val(filename), path("${video_file.baseName}_compressed.mp4"), emit: files

    script:
    """
    ffmpeg -i ${video_file} -c:v libx264 -pix_fmt yuv420p -preset veryfast -crf 23 -g 3000 -f mp4 ${video_file.baseName}_compressed.mp4
    """
}

/**
 * Uploads a file to Cloudian using rclone.
 *
 * @param tuple
 *  - result_file The path to the result file
 *  - publish_filename The desired publish filename
 * @param rclone_prefix The rclone remote prefix where files are to be uploaded
 * @param rclone_config The rclone config file that provides remote cloudian authentication
 */
process PUT_DATA_TO_CLOUDIAN {
    container "/projects/kumar-lab/meta/images/mouse-tracking-runtime/rclone/latest.sif"
    cpus 1
    time 10.m
    memory 1.GB
    array 200
    queue 'xfer'
    clusterOptions '-q xfer'
    maxForks 2
    errorStrategy 'ignore'
    
    input:
    tuple path(result_file), val(publish_filename)
    val rclone_prefix
    path rclone_config

    script:
    """
    rclone copy --config=${rclone_config} --transfers=1 --copy-links --include ${result_file} . ${rclone_prefix}${publish_filename}
    """
}

/**
 * Main workflow to retrieve data, compress it, and send it to a different remote server.
 *
 * params.input_batch is a list of files where each line contains a remote video to compress.
 * The pipeline splits this file by line and submits individual dependent jobs that run the following 3 steps:
 * 1. Retrieve the remote video data from dropbox to local compute
 * 2. Compress the video data
 * 3. Push the compressed video to cloudian remote storage
 */
workflow {
    input_files = channel.fromPath(params.input_batch)
    single_line_files = input_files.splitText(by: 1, file: true)
    local_videos = GET_DATA_FROM_DROPBOX(single_line_files, params.dropbox_prefix, params.dropbox_config).remote_files
    // `local_videos` is a text file containing the paths to the retrieved files. This is because we need folders (which are removed from Paths by nextflow)
    // Convert that to a channel
    local_video_channel = local_videos.splitText().map { line -> tuple(line.trim(), file(line.trim())) }
    compressed_videos = COMPRESS_VIDEO_CRF(local_video_channel).files

    sync_names = compressed_videos.map { original_file, video ->
        tuple(video, "${file(original_file).getParent().toString().replaceAll(".*retrieved_files/", "/")}")
    }
    PUT_DATA_TO_CLOUDIAN(sync_names, params.cloudian_prefix, params.cloudian_config)

}