include { FILTER_LOCAL_BATCH;
          URLIFY_FILE } from "${projectDir}/nextflow/modules/utils"
include { CHECK_GLOBUS_AUTH;
          FILTER_UNPROCESSED_GLOBUS;
          FILTER_UNPROCESSED_DROPBOX;
          TRANSFER_GLOBUS;
          GET_DATA_FROM_DROPBOX;
        } from "${projectDir}/nextflow/modules/remote_io"

workflow PREPARE_DATA {
    take:
    in_video_file
    location

    main:
    {

        if (location == "local")
            input_batch = FILTER_LOCAL_BATCH(in_video_file, params.ignore_invalid_inputs, params.filter_processed, params.pubdir).process_filelist
            video_file_batch = Channel.fromPath(video_file_batch)
        else if (location == "dropbox")
            in_video_list = FILTER_UNPROCESSED_DROPBOX(in_video_file, params.dropbox_prefix).unprocessed_files
            video_file_batch = GET_DATA_FROM_DROPBOX(in_video_file, params.dropbox_prefix).remote_files
        else if (location == "globus")
            CHECK_GLOBUS_AUTH()
            in_video_list = FILTER_UNPROCESSED_GLOBUS(params.globus_remote_endpoint, in_video_file).unprocessed_files
            globus_out_folder = TRANSFER_GLOBUS(params.globus_remote_endpoint, params.globus_compute_endpoint, in_video_list).globus_folder
            video_file_batch = Channel.fromPath(file(globus_out_folder).text)
        else error "${location} is invalid, specify local, dropbox, or globus"

        // Files should be appropriately URLified to avoid collisions within the pipeline
        input_video_channel = URLIFY_FILE(video_file_batch, params.path_depth).file
    }

    emit:
    input_video_channel
}
