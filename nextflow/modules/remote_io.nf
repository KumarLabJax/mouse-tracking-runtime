/**
 * This module contains process definitions for remote I/O operations.
 */

/**
 * Checks if the user has a valid Globus authentication session.
 *
 * @param globus_endpoint The Globus endpoint to check authentication against
 *
 * @exception If the user is not authenticated, the process will exit with an error.
 */
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

/**
 * Filters a list of video files to only those that do not have corresponding processed pose files on Globus.
 *
 * @param globus_endpoint The Globus endpoint where processed pose files are stored
 * @param test_files A file containing a list of video files to check
 *
 * @return unprocessed_files A file containing a list of video files that do not have a corresponding pose file.
 */
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

/**
 * Filters a list of video files to only those that do not have corresponding processed pose files on Dropbox.
 *
 * @param dropbox_prefix The rclone remote prefix where processed pose files are stored
 * @param test_files A file containing a list of video files to check
 *
 * @return unprocessed_files A file containing a list of video files that do not have corresponding processed pose files
 */
process FILTER_UNPROCESSED_DROPBOX {
    label "rclone"
    label "dropbox"

    input:
    path test_files
    val dropbox_prefix
    path rclone_config

    output:
    path "unprocessed_files.txt", emit: unprocessed_files

    script:
    """
    #!/bin/bash

    touch unprocessed_files.txt
    while read test_file; do
        test_pose=\${test_file/.*}_pose_est_v6.h5
        rclone ls --config=${rclone_config} ${dropbox_prefix}\${test_pose} > /dev/null 2>&1
        if [[ \$? != 0 ]]; then
            echo \$test_file >> unprocessed_files.txt
        fi
    done < ${test_files}
    exit 0
    """
}

/**
 * Transfers files between 2 Globus endpoints.
 *
 * @param globus_src_endpoint The source Globus endpoint to transfer files from
 * @param globus_dst_endpoint The destination Globus endpoint to transfer files to
 * @param files_to_transfer A file containing a list of files to transfer
 *
 * @return globus_folder A file containing the path to this tasks folder.
 */
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

/**
 * Retrieves files from Dropbox using rclone.
 *
 * @param files_to_transfer A file containing a list of files to transfer
 * @param dropbox_prefix The rclone remote prefix where files are stored
 *
 * @return remote_files A file containing a list of the retrieved files with full paths.
 */
process GET_DATA_FROM_DROPBOX {
    label "rclone"
    label "dropbox"
    
    input:
    path files_to_transfer
    val dropbox_prefix
    path rclone_config

    output:
    path "fetched_files.txt", emit: remote_files

    script:
    """
    echo ${dropbox_prefix}
    rclone copy --config=${rclone_config} --transfers=1 --include-from ${files_to_transfer} ${dropbox_prefix} retrieved_files/.
    find \$(pwd)/retrieved_files/ -type f > fetched_files.txt
    """
}

/**
 * Uploads a file to Dropbox using rclone.
 *
 * @param file_to_upload The file to be uploaded
 * @param tuple
 *  - result_file The path to the result file
 *  - publish_filename The desired publish filename
 * @param dropbox_prefix The rclone remote prefix where files are to be uploaded
 */
process PUT_DATA_TO_DROPBOX {
    label "rclone"
    label "dropbox"
    
    input:
    path file_to_upload
    tuple path(result_file), val(publish_filename)
    val dropbox_prefix
    path rclone_config

    script:
    """
    rclone copy --config=${rclone_config} --transfers=1 ${result_file} ${dropbox_prefix}/${publish_filename}
    """
}
