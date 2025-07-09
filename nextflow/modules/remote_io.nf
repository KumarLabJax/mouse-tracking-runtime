process CHECK_GLOBUS_AUTH {
    label "globus"
    
    input:
    val globus_endpoint

    script:
    // TODO:
    // If the command fails, globus will print a message to re-authenticate
    // This message should be sent to the user via email.
    """
    globus ls ${globus_endpoint}:/
    if [[ \$? != 0 ]]; then
        echo "Globus authentication failed. Please re-authenticate."
        exit 1
    fi
    """

    // TODO: This check could be improved.
    // "globus session show -F json" can return a json containing auth_time
    // But this needs to be parsed and compared with the endpoint expiration
}

process TRANSFER_GLOBUS {
    label "globus"
    
    input:
    val globus_src_endpoint
    val globus_dst_endpoint
    val video_filename

    output:
    path video_file

    script:
    // Globus is asynchronous, so we need to capture the task and wait.
    """
    id=$(globus transfer --jq "task_id" --format=UNIX ${globus_src_endpoint}:/${video_filename} ${globus_dst_endpoint}:/${video_filename})
    globus task wait --polling-interval=10 \$id
    """
}

process GET_DATA_FROM_DROPBOX {
    label "rclone"
    label "dropbox"
    
    input:
    val video_filename

    output:
    path ${video_file.baseName}, emit: video_file

    script:
    """
    #!/bin/bash

    rclone ls ${DROPBOX_PREFIX}/\$video_filename > ./video_file_remote_stats.txt"
    h5_filename=${video_file.baseName}_pose_est_v6.h5
    rclone ls "${DROPBOX_PREFIX}/\${h5_filename}"
    if [[ \$? == 0 ]]; then
        echo "File already processed. Skipping."
        return 1
    fi
    required_space=\$(awk '{print \$1}' ./video_file_remote_stats.txt)
    available_space=\$(df . | awk '{ print \$4 }' | tail -n 1)
    if [[ $required_space -gt $available_space ]]; then
        echo "Not enough space to download file. Exiting."
        return 1
    fi
    rclone copy ${DROPBOX_PREFIX}/$video_filename .
    """
}

process PUT_DATA_TO_DROPBOX {
    label "rclone"
    label "dropbox"
    
    input:
    path file_to_upload
    val folder_name

    script:
    """
    rclone copy $file_to_upload ${DROPBOX_PREFIX}/$folder_name/.
    """
}