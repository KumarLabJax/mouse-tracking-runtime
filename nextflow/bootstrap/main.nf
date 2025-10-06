nextflow.enable.dsl=2

/*
 * This bootstrap workflow generates JABS classifiers and their associated metadata.
 * It produces versioned, content-hashed artifacts and a Nextflow configuration
 * file that can be used by the main analysis pipeline.
 */

/**
 * Creates a version directory and a JSON config file with metadata about the build.
 */
process CREATE_VERSION_CONFIG {
    tag "config_${params.jabs_version}"
    label 'cpu'
    publishDir "${params.classifier_base_path}/${params.jabs_version}", mode: 'copy', overwrite: true

    output:
    path "jabs.config.json"

    script:
    """
    printf '{
' > "jabs.config.json"
    printf '  "jabs_version": "${params.jabs_version}",
' >> "jabs.config.json"
    printf '  "creation_timestamp_utc": "%s"
' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "jabs.config.json"
    printf '}
' >> "jabs.config.json"
    """
}

process CALCULTE_PROJECT_WINDOW_SIZES {
    label 'cpu'

   input:
    val(project_folder_name)

    output:
    tuple val(project_folder_name), env('WINDOW_SIZES'), emit: project_with_windows

    script:
    def project_path = "${params.classifier_project_folders}/${project_folder_name}"
    def project_file = "${project_path}/jabs/project.json"
    """
    export WINDOW_SIZES=\$(jq -r '[.behavior[] | .window_size] | unique | map("-w \\(.)") | join(" ")' ${project_file})
    """
}

process INIT_JABS_PROJECTS {
    label 'jabs_classify'
    label 'highcpu'

    input:
    tuple val(project_folder_name), val(window_sizes)

    output:
    val(project_folder_name), emit: initialized_project

    script:
    def project_path = "${params.classifier_project_folders}/${project_folder_name}"
    """
    jabs-init "${project_path}" ${window_sizes} -p 16
    """
}

process EXPORT_TRAINING_DATA {
    tag "export_${behavior_path}"
    label 'jabs_classify'
    label 'cpu'

    publishDir path: "${params.classifier_base_path}/${params.jabs_version}/${behavior_path}",
               mode: 'copy'

    input:
    tuple val(behavior_name), val(behavior_path), val(project_folder_name)

    output:
    tuple val(behavior_name), val(behavior_path), val(project_folder_name), path("*.h5"), env('HASH'), emit: h5_file_with_hash
    path "*.h5.manifest.json"

    script:
    def project_path = "${params.classifier_project_folders}/${project_folder_name}"
    """
    # 1. Export to a local file
    jabs-cli export-training --behavior "${behavior_name}" --outfile "training.h5"  "${project_path}"

    # 2. Calculate hash and export it
    export HASH=\$(sha256sum "training.h5" | awk '{ print \$1 }')

    # 3. Create the content-addressed files
    mv training.h5 "\${HASH}.h5"

    # 4. Create the manifest file
    cat > "\${HASH}.h5.manifest.json" <<EOF
{
  "behavior": "${behavior_name}",
  "jabs_version": "${params.jabs_version}",
  "file_hash_sha256": "\${HASH}",
  "timestamp_utc": "\$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "source_jabs_project": "${project_folder_name}"
}
EOF
    """
}

process TRAIN_CLASSIFIER {
    tag "train_${behavior_path}"
    label 'jabs_classify'
    label 'cpu'

    publishDir path: "${params.classifier_base_path}/${params.jabs_version}/${behavior_path}",
               mode: 'copy'

    input:
    tuple val(behavior_name), val(behavior_path), val(project_folder_name), path(training_h5), val(training_hash)

    output:
    tuple val(behavior_name), val(behavior_path), path("*.pickle"), env('HASH'), emit: classifier_file
    path "*.pickle.manifest.json"

    script:
    """
    # 1. Train the classifier
    jabs-classify train "${training_h5}" "classifier.pickle"

    # 2. Calculate hash of the new classifier
    export HASH=\$(sha256sum "classifier.pickle" | awk '{ print \$1 }')

    # 3. Create content-addressed files
    mv classifier.pickle "\${HASH}.pickle"

    # 4. Create the classifier manifest file
    cat > "\${HASH}.pickle.manifest.json" <<EOF
{
  "behavior": "${behavior_name}",
  "jabs_version": "${params.jabs_version}",
  "file_hash_sha256": "\${HASH}",
  "timestamp_utc": "\$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "source_training_data_hash": "${training_hash}",
  "training_file": "${training_h5}",
  "training_manifest": "${training_h5}.manifest.json",
}
EOF
    """
}

/**
 * Generates a Nextflow configuration file that maps behaviors to their
 * generated classifier artifacts for the main analysis pipeline.
 */
process GENERATE_PIPELINE_CONFIG {
    tag "generate_config"
    label 'cpu'
    publishDir "${params.classifier_base_path}/${params.jabs_version}", mode: 'copy', overwrite: true

    input:
    val collected_behaviors // list of [behavior_name, behavior_path, classifier_hash]

    output:
    path "generated_classifiers.config"

    script:
    def behaviors_json  = groovy.json.JsonOutput.toJson(collected_behaviors)
    def classifiers_json = groovy.json.JsonOutput.toJson(params.single_mouse_classifiers)
    """
#!/usr/bin/env python3
import json
from pathlib import Path

behaviors = json.loads('''${behaviors_json}''')

classifiers = json.loads('''${classifiers_json}''')

with open("generated_classifiers.config", "w") as f:
    f.write("params {\\n")
    f.write("    single_mouse_classifiers = [\\n")
    
    # Iterate through behaviors in triples (behavior_name, behavior_path, classifier_file)
    for row in behaviors:
        b_name, b_path, c_hash = row
        details = classifiers[b_name]
    
        classifier_path = Path("${params.classifier_base_path}/${params.jabs_version}/") / b_path / Path(c_hash + ".pickle")
        
        f.write(f"        '{b_name}': [\\n")
        f.write(f"            classifier_path: '{classifier_path}',\\n")
        f.write(f"            stitch_value: {details['stitch_value']},\\n")
        f.write(f"            filter_value: {details['filter_value']}\\n")
        f.write(f"        ],\\n")
    
    f.write("    ]\\n")
    f.write("}\\n")
"""
}


/**
 * Main workflow for generating classifiers and the pipeline config.
 */
workflow {
    main:
    CREATE_VERSION_CONFIG()

    // Create channel of behaviors with their details
    classifier_ch = Channel.from(params.single_mouse_classifiers.collect { k, v -> [k, v] })
    
    // Map to include behavior_path
    behavior_projects_ch = classifier_ch.map { behavior_name, details ->
        def behavior_path = behavior_name.replaceAll(' ', '_').replaceAll('[()]', '')
        tuple(behavior_name, behavior_path, details.project_folder_name)
    }

    // Extract unique project folders
    unique_projects_ch = classifier_ch
        .map { behavior_name, details -> details.project_folder_name }
        .unique()

    // Calcualte project windows outside of container so we have access to jq
    project_with_windows = CALCULTE_PROJECT_WINDOW_SIZES(unique_projects_ch)

    // Initialize each unique project
    initialized_projects_ch = INIT_JABS_PROJECTS(project_with_windows)

    // Create a value channel from initialized projects for joining
    initialized_projects_val = initialized_projects_ch.initialized_project
        .collect()
        .map { projects -> 
            projects.collectEntries { [it, true] }
        }

    // Join behaviors with their initialized projects
    ready_behaviors_ch = behavior_projects_ch
        .combine(initialized_projects_val)
        .map { behavior_name, behavior_path, project_folder_name, project_map ->
            // Check if this behavior's project has been initialized
            if (project_map[project_folder_name]) {
                tuple(behavior_name, behavior_path, project_folder_name)
            }
        }

    exported_h5_ch = EXPORT_TRAINING_DATA(ready_behaviors_ch)

    trained_classifiers_ch = TRAIN_CLASSIFIER(exported_h5_ch.h5_file_with_hash)

    // Collect only the behavior names that were successfully trained
    trained_classifiers_ch.classifier_file
        .map { behavior_name, behavior_path, _classifier_file, hash -> tuple(behavior_name, behavior_path, hash) }
        .toList()
        .set{ collected_behaviors }

    GENERATE_PIPELINE_CONFIG(collected_behaviors)
}
