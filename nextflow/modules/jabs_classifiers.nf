process GENERATE_FEATURE_CACHE {
    // This process will correct pose pathing to a v6 file
    label "jabs_classify"

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

process PREDICT_CLASSIFIERS {
    label "jabs_classify"

    input:
    // Pose file must be of form "${video_file.baseName}_pose_est_v[0-9]+.h5"
    tuple path(in_pose), path(feature_cache)
    val classifiers

    output:
    tuple path(in_pose), path(feature_cache), path("${in_pose.baseName.replaceFirst(/_pose_est_v[0-9]+/, "")}_behavior.h5"), emit: files

    script:
    """
    for classifier in ${classifiers.join(' ')};
    do
        ln -s \${classifier} .
        jabs-classify classify --classifier \$(basename \${classifier}) --input-pose ${in_pose} --out-dir . --feature-dir .
    done
    """
}

process GENERATE_BEHAVIOR_TABLES {
    label "jabs_postprocess"

    input:
    tuple path(in_pose), path(feature_cache), path(classifier_file)
    val feature_files

    output:
    tuple path("${in_pose.baseName}/*_bouts.csv"), path("${in_pose.baseName}*_summaries.csv"), emit: files

    script:
    """
    behavior_command="--behavior ${feature_files.join(' --behavior ')}"
    mkdir -p ${in_pose.baseName}
    python3 /JABS-postprocess/generate_behavior_tables.py --project_folder . --feature_folder . --out_prefix ${in_pose.baseName} --out_bin_size 5 \${behavior_command}
    """
}

process PREDICT_HEURISTICS {
    label "jabs_postprocess"

    input:
    tuple path(in_pose), path(in_behavior), path(feature_cache)
    var heuristic_classifiers

    output:
    tuple path("${in_pose.baseName}/*_bouts.csv"), path("${in_pose.baseName}/*_summaries.csv"), emit: files

    script:
    """
    mkdir -p ${in_pose.baseName}
    for classifier in ${heuristic_classifiers};
    do
        python3 /JABS-postprocess/heuristic_classify.py --project_folder . --feature_folder . --behavior_config \${classifier} --out_prefix ${in_pose.baseName}/\${classifier} --out_bin_size 5
    done
    """
}