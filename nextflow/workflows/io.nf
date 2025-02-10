include { CHECK_FILE } from './../../nextflow/modules/utils'
include { GET_DATA_FROM_T2; PUT_DATA_TO_T2 } from './../../nextflow/modules/remote_io'
include { GET_DATA_FROM_DROPBOX; PUT_DATA_TO_DROPBOX } from './../../nextflow/modules/remote_io'

workflow PREPARE_DATA {
    take:
    in_video_file
    location

    main:
    {
        if (location == "local")
            video_file = CHECK_FILE(in_video_file).file
        else if (location == "dropbox")
            video_file = GET_DATA_FROM_DROPBOX(in_video_file).out.video_file
        // T2 retrieval not implemented yet, due to globus permission issue.
        // else if (location == "t2")
        // """
        //    GET_DATA_FROM_T2(${in_video_file})
        // """
        else error "${location} is invalid, specify either local or dropbox"
    }

    emit:
    video_file
}
