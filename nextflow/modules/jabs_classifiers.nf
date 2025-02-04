process GENERATE_FEATURE_CACHE {
    label "jabs_classify_0.18.0"

    input:
    tuple path(video_file), path(in_pose)

    output:
    tuple path(in_pose), path("${video_file.baseName}"), emit: files

    script:
    """

    """
}

process PREDICT_CLASSIFIERS {
    label "jabs_classify_0.18.0"

    input:
    tuple path(in_pose), path(feature_cache)
    var classifiers

    output:
    tuple path(in_pose), path(feature_cache), path("${in_pose.baseName}_behavior.h5"), emit: files

    script:
    """
    for classifier in ${classifiers};
    do
        if [ ! -f "${process.classifier_folder}\${classifier}${process.classifier_artifact_suffix}" ]; then
            echo "Classifier artifact not found: ${process.classifier_folder}\${classifier}${process.classifier_artifact_suffix}"
            exit 1
        fi

        jabs-classify --classifier ${process.classifier_folder}\${classifier}${process.classifier_artifact_suffix} --input-pose ${in_pose} --out-dir . --feature-cache ${feature_cache}
    done
    """
}

process GENERATE_BEHAVIOR_TABLES {
    label "jabs_postprocessing_0.18.0"

    input:
    tuple path(in_pose), path(in_behavior), path(feature_cache)
    val classifiers

    output:
    tuple path("${in_pose.baseName}/*_bouts.csv"), path("${in_pose.baseName}/*_summaries.csv"), emit: files

    script:
    // TODO: Add in classifiers by converting list to a string of "--behavior ${classifier}" for each classifier
    """
    
    mkdir -p ${in_pose.baseName}
    python3 /JABS-postprocess/generate_behavior_tables.py --project_folder . --feature_folder ${feature_cache} --out_prefix ${in_pose.baseName}/ --out_bin_size 5
    """
}

process PREDICT_HEURISTICS {
    label "jabs_postprocessing_0.18.0"

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