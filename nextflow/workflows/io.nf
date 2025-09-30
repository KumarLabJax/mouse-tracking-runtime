/**
 * This module contains workflows that manipulate inputs and outputs to pipelines.
 */
 
include { FILTER_LOCAL_BATCH;
          URLIFY_FILE;
          validateInputFile;
          validateInputFilelist } from "${projectDir}/nextflow/modules/utils"
include { CHECK_GLOBUS_AUTH;
          FILTER_UNPROCESSED_GLOBUS;
          FILTER_UNPROCESSED_DROPBOX;
          TRANSFER_GLOBUS;
          GET_DATA_FROM_DROPBOX;
        } from "${projectDir}/nextflow/modules/remote_io"

/**
 * Prepares data for the main pipeline. Steps include:
 * - Validating input files by extension
 * - [Optional] Filtering of already processed files
 * - Retrieval of files if remote
 * - URLifying file names to avoid collisions
 *
 * @param path in_video_file Text file containing list of files to process. One file per line.
 * @param val location The location of the input files (local, dropbox, or globus).
 * @param val skip_urlify If true, skips the URLification step. Some usages may already have URLified files.
 *
 * @return path file_processing_channel The channel containing the prepared input files.
 */
workflow PREPARE_DATA {
    take:
    in_video_file
    location
    skip_urlify

    main:
    // Validate input file extensions
    // TODO: This needs to be a file, not a list. Having the list here wrapping files as paths for filtering.
    // all_valid_files = validateInputFilelist(in_video_file, params.workflow)
    all_valid_files = file(in_video_file)

    if (location == "local") {
        file_batch = FILTER_LOCAL_BATCH(all_valid_files, params.ignore_invalid_inputs, params.filter_processed, params.pubdir).process_filelist
    } else if (location == "dropbox") {
        in_video_list = FILTER_UNPROCESSED_DROPBOX(all_valid_files, params.dropbox_prefix, params.rclone_config).unprocessed_files
        file_batch = GET_DATA_FROM_DROPBOX(in_video_list, params.dropbox_prefix, params.rclone_config).remote_files
    } else if (location == "globus") {
        CHECK_GLOBUS_AUTH()
        in_video_list = FILTER_UNPROCESSED_GLOBUS(params.globus_remote_endpoint, all_valid_files).unprocessed_files
        globus_out_folder = TRANSFER_GLOBUS(params.globus_remote_endpoint, params.globus_compute_endpoint, in_video_list).globus_folder
        file_batch = Channel.fromPath(file(globus_out_folder).text)
    } else {
        error "${location} is invalid, specify local, dropbox, or globus"
    }

    // Files should be appropriately URLified to avoid collisions within the pipeline
    if (skip_urlify) {
        file_processing_channel = file_batch.splitText().map { line -> file(line.trim()) }
    } else {
        file_processing_channel = URLIFY_FILE(file_batch.splitText().map { it.trim() }, params.path_depth).file
    }

    emit:
    file_processing_channel
}
