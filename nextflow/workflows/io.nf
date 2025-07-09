include { CHECK_FILE; URLIFY_FILE } from "${projectDir}/nextflow/modules/utils"
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
            video_file = CHECK_FILE(in_video_file).file
        // TODO: Change remote retrieval to be serialized to not DDOS the network
        else if (location == "dropbox")
            in_video_list = FILTER_UNPROCESSED_DROPBOX(in_video_file).unprocessed_files
            video_file = GET_DATA_FROM_DROPBOX(in_video_file).out.video_file
        else if (location == "globus")
            CHECK_GLOBUS_AUTH()
            in_video_list = FILTER_UNPROCESSED_GLOBUS(params.globus_remote_endpoint, in_video_file).unprocessed_files
            video_file = TRANSFER_GLOBUS(params.globus_remote_endpoint, params.globus_compute_endpoint, in_video_file).out.video_file
        else error "${location} is invalid, specify either local, dropbox, or globus"
        out_file = URLIFY_FILE(video_file, params.path_depth).file
    }

    emit:
    out_file
}
