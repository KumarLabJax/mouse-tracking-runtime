/**
 * Lazy nextflow module for creating files, useful for testing.
 *
 * @param file_name The name of the file to be created
 * @param file_content The content to be written to the file
 * @return A path to the created file
 */
process CREATE_FILE {
    label "r_util"

    input:
    val file_name
    val file_content

    output:
    path file_name, emit: created_file

    script:
    """
    echo "${file_content}" > ${file_name}
    sleep 10
    """
}

process FILTER_LOCAL_BATCH {
    label "r_util"

    input:
    path input_batch
    val ignore_invalid_inputs
    val filter_processed
    val search_dir

    output:
    path "files_to_process.txt", emit: process_filelist

    script:
    """
    touch files_to_process.txt
    while IFS="" read -r file; do
        if [[ ! -f "\${file}" && ${ignore_invalid_inputs} != "true" ]]; then
            echo "File does not exist: \${file}"
            exit 1
        else
            echo "\${file} exists, adding to process list."
            echo "\${file}" >> files_to_process.txt
        fi
    done < ${input_batch}
    
    if [[ ${filter_processed} == "true" ]]; then
        mv files_to_process.txt all_files.txt
        touch files_to_process.txt
        echo "Filtering out already processed files..."
        while IFS="" read -f file; do
            pose_file="${search_dir}/\${file/.*}_pose_est_v6.h5"
            if [[ -f "\${pose_file}" ]]; then
                echo "File \${file} already processed, skipping."
            else
                echo "\${file}" >> files_to_process.txt
            fi
        done < files_to_process.txt
    fi
    """
}

process VIDEO_TO_POSE {
    label "r_util"

    // Generates a dummy pose file such that the pipeline can start at any step
    input:
    path video_file

    output:
    tuple path(video_file), path("${video_file.baseName}_pose_est_v0.h5"), emit: files

    script:
    """
    touch "${video_file.baseName}_pose_est_v0.h5"
    sleep 10
    """
}

process URLIFY_FILE {
    label "r_util"

    // WARNING: This process will fail if depth > actual file depth
    input:
    val file_to_urlify
    val depth

    output:
    path "${file_to_urlify.split('/')[-1-depth..-1].join('%20')}", emit: file

    script:
    """
    ln -s ${file_to_urlify} "${file_to_urlify.split('/')[-1-depth..-1].join('%20')}"
    sleep 10
    """
}

process REMOVE_URLIFY_FIELDS {
    label "r_util"

    input:
    path urlified_file

    output:
    path "${urlified_file.baseName}_no_urls${urlified_file.extension}", emit: file

    script:
    """
    sed -e 's:%20:/:g' ${urlified_file} > "${urlified_file.baseName}_no_urls${urlified_file.extension}"
    sleep 10
    """
}

process MERGE_FEATURE_ROWS {
    label "r_util"

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
        tail -n+\$((${header_size}+1)) "\$feature_file" >> ${out_filename}.csv
    done
    sleep 10
    """
}

process MERGE_FEATURE_COLS {
    // Any environment with pandas installed should work here.
    label "tracking"
    label "r_util"

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
    label "r_util"

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
    sleep 10
    """
}

process ADD_COLUMN {
    label "r_util"

    input:
    path file_to_add_to
    val column_name
    val column_data

    output:
    path "${file_to_add_to.baseName}_with_${column_name}${file_to_add_to.extension}", emit: file

    script:
    """
    awk 'BEGIN {FS=OFS=","} NR==1 {print \$0, "${column_name}"} NR>1 {print \$0, "${column_data}"}' ${file_to_add_to} > ${file_to_add_to.baseName}_with_${column_name}${file_to_add_to.extension}
    sleep 10
    """
}

process DELETE_ROW {
    label "r_util"

    input:
    path file_to_delete_from
    val row_to_delete

    output:
    path "${file_to_delete_from.baseName}_no_${row_to_delete}${file_to_delete_from.extension}", emit: file

    script:
    """
    grep -v "${row_to_delete}" ${file_to_delete_from} > "${file_to_delete_from.baseName}_no_${row_to_delete}${file_to_delete_from.extension}"
    sleep 10
    """
}

process FEATURE_TO_LONG {
    // Any environment with pandas installed should work here.
    label "tracking"
    label "r_util"

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
    label "r_util"

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
    label "r_util"

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
    sleep 10
    """
}

process PUBLISH_RESULT_FILE {
    label "r_util"

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
    sleep 10
    """
}

process GET_WORKFLOW_VERSION {
    label "r_util"

    publishDir "${params.pubdir}", mode:'copy', overwrite: false

    output:
    val "${workflow.commitId ?: params.git_hash}", emit: version
    path "workflow_version.txt", emit: version_file

    script:
    """
    echo "nextflow_revision=${workflow.commitId ?: params.git_hash}" > workflow_version.txt
    echo "workflow_version=${workflow.manifest.version ?: 'UNSET'}" >> workflow_version.txt
    echo "git_head=${params.git_hash}" >> workflow_version.txt
    echo "date_run=\$(date +%F)" >> workflow_version.txt
    sleep 10
    """
}

process ADD_DUMMY_VIDEO {
    // Any environment with ffmpeg installed should work here.
    label "tracking"
    label "r_gen_vid"

    input:
    path pose_file
    val n_frames

    output:
    tuple path("${pose_file.baseName.replaceFirst(/_pose_est_v[0-9]+/, "")}.mp4"), path(pose_file), emit: files

    script:
    """
    ffmpeg -f lavfi -i color=size=480x480:rate=30:color=black -vframes "${n_frames}" "${pose_file.baseName.replaceFirst(/_pose_est_v[0-9]+/, "")}.mp4"
    """
}

/**
 * Validates input files based on specified criteria and pipeline type
 *
 * @param file_path The path to the file that needs validation
 * @param pipeline_type The type of pipeline being run (e.g. 'single-mouse', 'single-mouse-corrected-corners', etc.)
 * @return A boolean indicating if the file is valid and an error message if it's not
 */
def validateInputFile(String file_path, String pipeline_type) {
    def file = file(file_path)
    def valid_extensions = [
        'single-mouse': ['.avi', '.mp4'],
        'single-mouse-corrected-corners': ['.h5'],
        'single-mouse-v6-features': ['.h5'],
        'multi-mouse': ['.avi', '.mp4']
    ]
    
    // Check if pipeline type is valid
    if (!valid_extensions.containsKey(pipeline_type)) {
        return [false, "Invalid pipeline type: ${pipeline_type}. Expected one of: ${valid_extensions.keySet()}"]
    }
    
    def extension = file_path.substring(file_path.lastIndexOf('.'))
    
    // Check file extension against allowed extensions for pipeline type
    if (!valid_extensions[pipeline_type].contains(extension.toLowerCase())) {
        return [false, "Invalid file extension: ${extension}. For pipeline ${pipeline_type}, expected one of: ${valid_extensions[pipeline_type]}"]
    }
    
    return [true, ""]
}

/**
 * Subsets an input file list by the formats allowed for a specific pipeline type.
 *
 * @param in_file_list The path to the file that contains the list of intut files
 * @param pipeline_type The type of pipeline being run. See validateInputFile for valid types.
 * @return A list of valid file paths that match the allowed formats for the specified pipeline type.
 */
def validateInputFilelist(String in_file_list, String pipeline_type) {
    def all_valid_files = []
    def invalid_files = []
    def valid_files = []

    def batch_lines = file(in_file_list).text.readLines()

    // Validate each file in the batch
    batch_lines.each { file_path ->
        def (is_valid, error_message) = validateInputFile(file_path, pipeline_type)

        if (is_valid) {
            valid_files.add(file_path)
        } else {
            invalid_files.add([file_path, error_message])
        }
    }

    // Report any invalid files
    if (invalid_files.size() > 0) {
        println "The following files failed validation:"
        invalid_files.each { file_path, error_message ->
            println "  - ${error_message}"
        }

        if (!params.ignore_invalid_inputs) {
            println "Please check the input files and try again."
            println "If you want to ignore invalid inputs, please set the parameter ignore_invalid_inputs to true."
            System.exit(1)
        }

        // If all files are invalid, exit
        if (valid_files.size() == 0) {
            println "No valid files to process. Exiting."
            System.exit(1)
        }

        // Otherwise, continue with valid files and warn the user
        println "Continuing with ${valid_files.size()} valid files out of ${batch_lines.size()} total files."
    }

    all_valid_files.addAll(valid_files)

    if (all_valid_files.size() == 0){
        println "Missing any data to process, please assign either input_data or input_batch"
        System.exit(1)
    }

    return all_valid_files
}
