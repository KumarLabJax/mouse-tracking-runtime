process VIDEO_TO_POSE {
    // Generates a dummy pose file such that the pipeline can start at any step
    input:
    path video_file

    output:
    tuple path(video_file), path("${video_file.baseName}_pose_est_v0.h5"), emit: files

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

process MERGE_FEATURE_ROWS {
    input:
    path feature_files
    val out_filename
    val header_size

    output:
    path "${out_filename}.csv", emit: merged_features

    script:
    """
    head -n${header_size} ${feature_files[0]} > ${out_filename}.csv
    for feature_file in ${feature_files};
    do
        tail -n+\$((${header_size}+1)) \${feature_file} >> ${out_filename}.csv
    done
    """
}

process MERGE_FEATURE_COLS {
    // Any environment with pandas installed should work here.
    label "tracking"

    input:
    path feature_files
    val col_to_merge_on
    val out_filename

    output:
    path "${out_filename}.csv", emit: merged_features

    script:
    """
    #!/usr/bin/env python3

    import pandas as pd
    import functools
    file_list = [${feature_files.collect { "\"${it}\"" }.join(', ')}]
    read_data = [pd.read_csv(f).drop("Unnamed: 0", axis=1) for f in file_list]
    merged_data = functools.reduce(lambda left, right: pd.merge(left, right, on="${col_to_merge_on}"), read_data)
    merged_data.to_csv("${out_filename}.csv", index=False)
    """
}

process FILTER_QC_WITH_CORNERS {
    input:
    path(qc_file)

    output:
    path "${qc_file.baseName}_with_corners.txt", emit: with_corners
    path "${qc_file.baseName}_without_corners.txt", emit: without_corners

    script:
    """
    awk -F',' '
    NR==1 {
        for (i=1; i<=NF; i++) {
            f[\$i] = i
        }
    }
    {
        if (\$(f["corners_present"]) == "True") print \$(f["pose_file"])
    }
    ' ${qc_file} > "${qc_file.baseName}_with_corners.txt"
    
    awk -F',' '
    NR==1 {
        for (i=1; i<=NF; i++) {
            f[\$i] = i
        }
    }
    {
        if (\$(f["corners_present"]) == "False") print \$(f["pose_file"])
    }
    ' ${qc_file} > "${qc_file.baseName}_without_corners.txt"
    """
}
