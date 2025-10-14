/**
 * This module contains process definitions related to JABS classifiers.
 */

/**
 * Calculates features for JABS classifiers.
 *
 * @param tuple
 *  - video_file The input video file.
 *  - in_pose The input pose file.
 *
 * @return tuple files
 *  - Path to the original pose file.
 *  - Path to the generated feature cache directory.
 */
process GENERATE_FEATURE_CACHE {
    // This process will correct pose pathing to a v6 file
    label "jabs_classify"
    label "cpu"
    label "r_jabs_features"

    input:
    tuple path(video_file), path(in_pose)

    output:
    tuple path("${video_file.baseName}_pose_est_v6.h5"), path("features/${video_file.baseName}_pose_est_v6"), emit: files

    script:
    """
    if [ ! -f "${video_file.baseName}_pose_est_v6.h5" ]; then
        ln -s ${in_pose} "${video_file.baseName}_pose_est_v6.h5"
    fi
    mkdir -p ${video_file.baseName}
    for window_size in ${params.classifier_window_sizes.join(' ')};
    do
        jabs-features --pose-file "${video_file.baseName}_pose_est_v6.h5" --feature-dir features --pose-version 6 --window-size \${window_size} --use-cm-distances
    done
    """
}  

/**
 * Predicts behaviors using JABS classifiers.
 *
 * @param tuple
 *  - in_pose The input pose file.
 *  - feature_cache The directory containing the generated features.
 * @param classifiers A map of classifiers:
 *  - <classifier_name>: classifier parameter maps (not used by this process)
 *
 * @return tuple files
 *  - Path to the original pose file.
 *  - Path to the feature cache directory.
 *  - Path to the generated behavior file. All behavior predictions are stored in a single file.
 */
process PREDICT_CLASSIFIERS {
    label "jabs_classify"
    label "cpu"
    label "r_jabs_classify"

    input:
    // Pose file must be of form "${video_file.baseName}_pose_est_v[0-9]+.h5"
    tuple path(in_pose), path(feature_cache)
    val classifiers

    output:
    tuple path(in_pose), path(feature_cache), path("${in_pose.baseName.replaceFirst(/_pose_est_v[0-9]+/, "")}_behavior.h5"), emit: files

    script:
    """
    for classifier_path in ${classifiers.collect { _behavior, details -> details.classifier_path }.join(' ')};
    do
        jabs-classify classify --classifier "\${classifier_path}" --input-pose ${in_pose} --out-dir . --feature-dir .
    done
    """
}

/**
 * Generates behavior tables from pose file, feature file, and prediction file.
 *
 * @param tuple
 *  - in_pose The input pose file.
 *  - feature_cache The directory containing the generated features.
 *  - behavior_files The behavior prediction file.
 * @param classifiers A map of classifiers:
 *  - <classifier_name>: classifier parameter maps:
 *    - stitch_value: the gap size for stitching behavior bouts
 *    - filter_value: the minimum length for behavior bouts
 *
 * @return tuple files
 *  - Path to the generated behavior bout file.
 *  - Path to the generated behavior summary file.
 */
process GENERATE_BEHAVIOR_TABLES {
    label "jabs_postprocess"
    label "cpu"
    label "r_jabs_tablegen"

    input:
    tuple path(in_pose), path(feature_cache), path(behavior_files)
    val classifiers

    output:
    tuple path("${in_pose.baseName}*_bouts.csv"), path("${in_pose.baseName}*_summaries.csv"), emit: files

    script:
    def behaviorJson = groovy.json.JsonOutput.toJson([
        behaviors: classifiers.collect { entry ->
            [
                behavior: entry.key,
                stitch_gap: entry.value.stitch_value,
                min_bout_length: entry.value.filter_value
            ]
        }
    ])
    """
    echo '${behaviorJson}' > config.json
    jabs-postprocess generate-tables \
        --project-folder . \
        --feature-folder . \
        --out-prefix ${in_pose.baseName} \
        --out-bin-size 5 \
        --behavior-config ./config.json
    """
}

/**
 * Generates heuristic classifier predictions.
 *
 * @param tuple
 *  - in_pose The input pose file.
 *  - feature_cache The directory containing the generated features.
 * @param heuristic_classifiers A list of heuristic classifier configuration files.
 *
 * @return tuple files
 *  - Path to the generated bout file.
 *  - Path to the generated summary file.
 */
process PREDICT_HEURISTICS {
    label "jabs_postprocess"
    label "cpu"
    label "r_jabs_heuristic"

    input:
    // Pose file must be of form "${video_file.baseName}_pose_est_v[0-9]+.h5"
    tuple path(in_pose), path(feature_cache)
    val heuristic_classifiers

    output:
    tuple path("${in_pose.baseName}*_bouts.csv"), path("${in_pose.baseName}*_summaries.csv"), emit: files

    script:
    """
    for classifier in ${heuristic_classifiers.join(' ')};
    do
        jabs-postprocess heuristic-classify \
            --project-folder . \
            --feature-folder . \
            --behavior-config \${classifier} \
            --out-prefix ${in_pose.baseName} \
            --out-bin-size 5
    done
    """
}

/**
 * Converts a behavior summary table to features.
 *
 * @param in_summary_table The input behavior summary table.
 * @param bin_size The bin size for feature extraction.
 *
 * @return features The generated feature file.
 */
process BEHAVIOR_TABLE_TO_FEATURES {
    label "jabs_table_convert"
    label "cpu"
    label "r_jabs_table_convert"

    input:
    tuple path(in_summary_table), val(bin_size)

    output:
    path("${in_summary_table.baseName}_features_${bin_size}.csv"), emit: features

    script:
    """
    Rscript ${params.support_code_dir}behavior_summaries.R -f ${in_summary_table} -b ${bin_size} -o "${in_summary_table.baseName}_features_${bin_size}.csv"
    """
}

/**
* Aggregate bout tables by behavior across all videos.
*
* This process uses jabs-postprocess to merge tables by behavior,
* creating separate merged files for each behavior detected.
*
* @param bout_tables List of paths to bout tables to be merged.
*
* @return merged_bout_tables List of paths to merged bout tables, one per behavior.
* @return merge_log Path to the log file detailing the merge process.
*
* @publish ./results/merged_behavior_tables Merged behavior bout tables
* @publish ./results/merged_behavior_tables Merge log file
*/
process AGGREGATE_BOUT_TABLES {
    label "jabs_postprocess"
    label "cpu"
    label "r_jabs_table_convert"

    publishDir "${params.pubdir}/merged_behavior_tables", mode: 'copy'

    input:
    path bout_tables

    output:
    path("merged_*_bouts_merged.csv"), emit: merged_bout_tables
    path("merge_log.txt"), emit: merge_log

    script:
    """
    # Create a temporary directory for organizing tables
    mkdir -p table_staging
    
    # Copy all bout tables to staging directory
    for table in ${bout_tables}; do
        cp "\${table}" table_staging/
    done
    
    # Use jabs-postprocess to merge tables by behavior
    # This will automatically detect behaviors and create separate merged files for each
    echo "Starting behavior table merging..." > merge_log.txt
    echo "Input tables found:" >> merge_log.txt
    ls table_staging/*.csv >> merge_log.txt
    
    jabs-postprocess merge-multiple-tables \\
        --table-folder table_staging \\
        --table-pattern "*_bouts.csv" \\
        --output-prefix merged \\
        --overwrite \\
        2>&1 | tee -a merge_log.txt
    
    # Log completion
    echo "Merge completed. Output files:" >> merge_log.txt
    ls merged_*_bouts_merged.csv >> merge_log.txt 2>/dev/null || echo "No merged files generated" >> merge_log.txt
    """
}

