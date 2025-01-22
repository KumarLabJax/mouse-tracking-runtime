process GET_DATA_FROM_T2 {
    label "globus"
    
    input:
    val video_filename

    output:
    path video_file

    script:
    """
    echo "Not implemented yet!"
    exit 1
    """
}

process PUT_DATA_TO_T2 {
    label "globus"
    
    input:
    path file_to_upload
    val folder_name

    script:
    """
    echo "Not implemented yet!"
    exit 1
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
    !/bin/bash

    rclone ls \$DROPBOX_PREFIX/\$video_filename > \$WORK_DIR/video_file_remote_stats.txt"
    h5_filename=${video_file.baseName}_pose_est_v6.h5
    rclone ls "\$DROPBOX_PREFIX/\${h5_filename}"
    if [[ \$? == 0 ]]; then
        echo "File already processed. Skipping."
        return 1
    fi
    required_space=\$(awk '{print \$1}' $WORK_DIR/video_file_remote_stats.txt)
    available_space=\$(df \$WORK_DIR | awk '{ print \$4 }' | tail -n 1)
    if [[ $required_space -gt $available_space ]]; then
        echo "Not enough space to download file. Exiting."
        return 1
    fi
    rclone copy $DROPBOX_PREFIX/$video_filename $WORK_DIR
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
    rclone copy $file_to_upload $DROPBOX_PREFIX/$folder_name/.
    """
}