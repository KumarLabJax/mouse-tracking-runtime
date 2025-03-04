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

process URLIFY_FILE {
    // WARNING: This process will fail if depth > actual file depth
    input:
    val file_to_urlify
    val depth

    output:
    path "${file_to_urlify.split('/')[-1-depth..-1].join('%20')}", emit: file

    script:
    """
    ln -s ${file_to_urlify} "${file_to_urlify.split('/')[-1-depth..-1].join('%20')}"
    """
}

process REMOVE_URLIFY_FIELDS {
    input:
    path urlified_file

    output:
    path "${urlified_file.baseName}_no_urls${urlified_file.extension}", emit: file

    script:
    """
    sed -e 's:%20:/:g' ${urlified_file} > "${urlified_file.baseName}_no_urls${urlified_file.extension}"
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
    feature_files_array=(${feature_files.collect { element -> "\"${element.toString()}\"" }.join(' ')})
    head -n${header_size} "\${feature_files_array[0]}" > ${out_filename}.csv
    for feature_file in "\${feature_files_array[@]}";
    do
        tail -n\$((${header_size}+1)) "\$feature_file" >> ${out_filename}.csv
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

process ADD_COLUMN {
    input:
    path file_to_add_to
    val column_name
    val column_data

    output:
    path "${file_to_add_to.baseName}_with_${column_name}${file_to_add_to.extension}", emit: file

    script:
    """
    awk 'BEGIN {FS=OFS=","} NR==1 {print \$0, "${column_name}"} NR>1 {print \$0, "${column_data}"}' ${file_to_add_to} > ${file_to_add_to.baseName}_with_${column_name}${file_to_add_to.extension}
    """
}

process DELETE_ROW {
    input:
    path file_to_delete_from
    val row_to_delete

    output:
    path "${file_to_delete_from.baseName}_no_${row_to_delete}${file_to_delete_from.extension}", emit: file

    script:
    """
    grep -v "${row_to_delete}" ${file_to_delete_from} > "${file_to_delete_from.baseName}_no_${row_to_delete}${file_to_delete_from.extension}"
    """
}

process FEATURE_TO_LONG {
    // Any environment with pandas installed should work here.
    label "tracking"

    input:
    path feature_file
    val id_col

    output:
    path "${feature_file.baseName}_long.csv", emit: long_file

    script:
    """
    #!/usr/bin/env python3

    import pandas as pd
    read_data = pd.read_csv("${feature_file.toString()}")
    melted_data = pd.melt(read_data, id_vars="${id_col}", var_name="feature_name", value_name="value")
    melted_data.to_csv("${feature_file.baseName.toString()}_long.csv", index=False)
    """
}

process LONG_TO_WIDE {
    // Any environment with pandas installed should work here.
    label "tracking"

    input:
    path long_file
    val id_col
    val feature_col
    val value_col

    output:
    path "${long_file.baseName}_wide.csv", emit: wide_file

    script:
    """
    #!/usr/bin/env python3

    import pandas as pd
    read_data = pd.read_csv("${long_file.toString()}")
    wide_data = read_data.pivot(index="${id_col}", columns="feature_name", values="value").reset_index()
    wide_data.to_csv("${long_file.baseName.toString()}_wide.csv", index=False)
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
        if [ \$(dirname ${publish_filename}) != "" ]; then
            mkdir -p \$(dirname ${publish_filename})
        fi
        ln -s \$(pwd)/${result_file} ${publish_filename}
    fi
    """
}

process GET_WORKFLOW_VERSION {
    output:
    val "${workflow.commitId ?: 'N/A'}", emit: version

    script:
    """
    echo "Workflow version: ${workflow.commitId ?: 'N/A'}"
    """
}

process ADD_DUMMY_VIDEO {
    // Any environment with ffmpeg installed should work here.
    label "tracking"

    input:
    path pose_file
    val n_frames

    output:
    tuple(path("${pose_file.baseName.replaceFirst(/_pose_est_v[0-9]+/, "")}.mp4"), pose_file), emit: files

    script:
    """
    ffmpeg -f lavfi -i color=size=480x480:rate=30:color=black -vframes "${n_frames}" "${pose_file.baseName.replaceFirst(/_pose_est_v[0-9]+/, "")}.mp4"
    """
}
