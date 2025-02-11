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
    file_list = [${feature_files.collect { element -> "\"${element.toString()}\"" }.join(', ')}]
    read_data = [pd.read_csv(f) for f in file_list]
    read_data = [x.drop("Unnamed: 0", axis=1) if "Unnamed: 0" in x.columns else x for x in read_data]
    merged_data = functools.reduce(lambda left, right: pd.merge(left, right, on="${col_to_merge_on}"), read_data)
    merged_data.to_csv("${out_filename}.csv", index=False)
    """
}

process SELECT_COLUMNS {
    input:
    path(qc_file)
    val key_1
    val key_2

    output:
    path "${qc_file.baseName}_${key_1}_${key_2}.csv", emit: csv_file

    script:
    """
    awk -F',' '
    NR==1 {
        for (i=1; i<=NF; i++) {
            f[\$i] = i
        }
    }
    {
        print \$(f["${key_1}"]), \$(f["${key_2}"])
    }
    ' OFS=',' ${qc_file} > "${qc_file.baseName}_${key_1}_${key_2}.csv"
    """
}

process SUBSET_PATH_BY_VAR {
    input:
    path all_files
    val subset_files
    val dir

    output:
    path("${dir}/*"), emit: subset_files

    script:
    """
    mkdir ${dir}
    for file in ${subset_files}
    do
        ln -s \$(pwd)/\${file} ${dir}/\$(basename \${file})
    done
    """
}

process PUBLISH_RESULT_FILE {
    publishDir "${params.pubdir}", mode:'copy'

    input:
    tuple path(result_file), val(publish_filename)

    output:
    path(publish_filename), emit: published_file
    
    script:
    """
    if [ ! -f ${publish_filename} ]; then
        ln -s ${result_file} ${publish_filename}
    fi
    """
}