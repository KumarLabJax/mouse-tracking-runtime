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
    path "unprocessed_files.txt", emit: unprocessed_files

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
    val dropbox_prefix

    output:
    path "unprocessed_files.txt", emit: unprocessed_files

    script:
    """
    #!/bin/bash

    touch unprocessed_files.txt
    while read test_file; do
        test_pose=\${test_file/.*}_pose_est_v6.h5
        rclone ls ${dropbox_prefix}\${test_pose} > /dev/null 2>&1
        if [[ \$? != 0 ]]; then
            echo \$test_file >> unprocessed_files.txt
        fi
    done < ${test_files}
    exit 0
    """
}

process TRANSFER_GLOBUS {
    label "globus"
    
    input:
    val globus_src_endpoint
    val globus_dst_endpoint
    path files_to_transfer

    output:
    path "globus_cache_folder.txt", emit: globus_folder

    script:
    // Globus is asynchronous, so we need to capture the task and wait.
    """
    while read line; do
        line_space_escaped=\$(echo \$line | sed 's: :\\ :g')
        echo \${line_space_escaped} \${line_space_escaped} >> batch_to_from.txt
    done < ${files_to_transfer}
    id=\$(globus transfer --jq "task_id" --format=UNIX --batch batch_to_from.txt ${globus_src_endpoint} ${globus_dst_endpoint})
    while true; do
        globus task wait --timeout 60 --timeout-exit-code 2 \$id
        # Task succeeded
        if [[ \$? == 0 ]]; then
            break
        # Task failed
        elif [[ \$? == 1 ]]; then
            echo "Globus transfer failed."
            exit 1
        # Timeout, still running. Figure out if something is wrong.
        elif [[ \$? == 2 ]]; then
            # To get all the task info:
            # globus task show --format=UNIX \$id > globus_task_info.txt
            fault_count=\$(globus task show --format=UNIX -jq "faults" \$id)
            if [[ \$fault_count -gt 0 ]]; then
                echo "Globus transfer failed with faults."
                globus task cancel \$id
                exit 1
            fi
        fi
    done
    echo \${pwd} > globus_cache_folder.txt
    """
}

process GET_DATA_FROM_DROPBOX {
    label "rclone"
    label "dropbox"
    
    input:
    path files_to_transfer
    val dropbox_prefix

    output:
    path "fetched_files.txt", emit: remote_files

    script:
    """
    echo ${dropbox_prefix}
    rclone copy --transfers=1 --include-from ${files_to_transfer} ${dropbox_prefix} retrieved_files/.
    find \$(pwd)/retrieved_files/ -type f > fetched_files.txt
    """
}

process PUT_DATA_TO_DROPBOX {
    label "rclone"
    label "dropbox"
    
    input:
    path file_to_upload
    tuple path(result_file), val(publish_filename)
    val dropbox_prefix

    script:
    """
    rclone copy --transfers=1 ${result_file} ${dropbox_prefix}/${publish_filename}
    """
}
