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

process FILTER_UNPROCESSED_GLOBUS {
    label "globus"

    input:
    val globus_endpoint
    path test_files

    output:
    path "unprocessed_files.txt", emit unprocessed_files

    script:
    """
    touch unprocessed_files.txt
    while read test_file; do
        test_pose=\${test_file/.*}_pose_est_v6.h5
        globus ls ${globus_endpoint}:/\${test_pose} > /dev/null 2>&1
        if [[ \$? != 0 ]]; then
            echo \$test_file >> unprocessed_files.txt
        fi
    done < ${test_files}
    """
}

process FILTER_UNPROCESSED_DROPBOX {
    label "rclone"
    label "dropbox"

    input:
    path test_files

    output:
    path "unprocessed_files.txt", emit unprocessed_files

    script:
    """
    touch unprocessed_files.txt
    while read test_file; do
        test_pose=\${test_file/.*}_pose_est_v6.h5
        rclone ls ${DROPBOX_PREFIX}/\${test_pose} > /dev/null 2>&1
        if [[ \$? != 0 ]]; then
            echo \$test_file >> unprocessed_files.txt
        fi
    done < ${test_files}
    """
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
    path "${video_filename}", emit: video_file

    script:
    """
    rclone copy ${DROPBOX_PREFIX}/${video_filename} ./${video_filename}
    """
}

process PUT_DATA_TO_DROPBOX {
    label "rclone"
    label "dropbox"
    
    input:
    path file_to_upload
    tuple path(result_file), val(publish_filename)

    script:
    """
    rclone copy ${result_file} ${DROPBOX_PREFIX}/${publish_filename}
    """
}
