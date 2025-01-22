process VIDEO_TO_POSE {
    // Generates a dummy pose file such that the pipeline can start at any step
    input:
    path video_file

    output:
    path "${video_file.baseName}_pose_est_v0.h5", emit: pose_file;

    script:
    """
    touch "${video_file.baseName}_pose_est_v0.h5"
    """
}

process CHECK_FILE {
    input:
    val file_to_check

    output:
    val file_to_check, emit: file
    // path "file(${file_to_check})", emit: file
    // val !file("${file_to_check}").exists(), emit: file_exists

    script:
    """
    echo "Checking ${file_to_check}"
    if [ -f "${file_to_check}" ]; then
        echo "File exists"
    else
        echo "File does not exist"
        exit 1
    fi
    """
}