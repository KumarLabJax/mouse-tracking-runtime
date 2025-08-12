/**
 * This module contains process definitions for generating flexibility metrics.
 * These metrics were published in https://doi.org/10.1038/s43587-022-00266-0
 */

/**
 * Generates flexibility index metrics from pose data.
 *
 * @param tuple
 *  - video_file The input video file.
 *  - pose_file The input pose file.
 *
 * @return tuple files
 *  - Path to the generated angles CSV file.
 *  - Path to the generated dAC CSV file.
 *  - Path to the generated dB CSV file.
 *  - Path to the generated flexdexraw CSV file.
 */
process GENERATE_FLEXIBILITY_INDEX {
    label "frailty"
    label "cpu"
    label "r_flexibility"

    input:
    tuple path(video_file), path(pose_file)

    output:
    path("${video_file.baseName}_aABC.csv"), emit: angles
    path("${video_file.baseName}_dAC.csv"), emit: dist_ac
    path("${video_file.baseName}_dB.csv"), emit: dist_b
    path("${video_file.baseName}_flexdexraw.csv"), emit: flexraw

    script:
    // Script ingests a batch file of videos and presumes that there is an associated _pose_est_v2.h5 file for each video
    // This pipeline does not guarantee that the pose file is named that
    // Batch file also only works with .avi extension (hardcoded)
    """
    if [ ! -f ${video_file.baseName}_pose_est_v2.h5 ]; then
        ln -s ${pose_file} ${video_file.baseName}_pose_est_v2.h5
    fi
    find . -name "*_pose_est_v2.h5" > batch.txt
    sed -i 's:_pose_est_v2.h5:.avi:' batch.txt
    python ${params.vfi_code_dir}/flexindex.py --input-root-dir ./ --output-root-dir ./ --video-file-list batch.txt
    rm batch.txt

    mv aABC.csv ${video_file.baseName}_aABC.csv
    mv dAC.csv ${video_file.baseName}_dAC.csv
    mv dB.csv ${video_file.baseName}_dB.csv
    mv flexdexraw.csv ${video_file.baseName}_flexdexraw.csv
    """
}

/**
 * Generates rear paw width metrics from pose data.
 *
 * @param tuple
 *  - video_file The input video file.
 *  - pose_file The input pose file.
 *
 * @return rearpaw Path to the generated rear paw width CSV file.
 * @return rearpawsave Path to the generated rear paw save CSV file.
 */
process GENERATE_REAR_PAW_WIDTH {
    label "frailty"
    label "cpu"
    label "r_rearpaw"

    input:
    tuple path(video_file), path(pose_file)

    output:
    path("${video_file.baseName}_rearpaw.csv"), emit: rearpaw
    path("${video_file.baseName}_rearpawsave.csv"), emit: rearpawsave

    script:
    // Script ingests a batch file of videos and presumes that there is an associated _pose_est_v2.h5 file for each video
    // This pipeline does not guarantee that the pose file is named that
    // Batch file also only works with .avi extension (hardcoded)
    """
    if [ ! -f ${video_file.baseName}_pose_est_v2.h5 ]; then
        ln -s ${pose_file} ${video_file.baseName}_pose_est_v2.h5
    fi
    find . -name "*_pose_est_v2.h5" > batch.txt
    sed -i 's:_pose_est_v2.h5:.avi:' batch.txt
    python ${params.vfi_code_dir}/rearpawwidths.py --input-root-dir ./ --output-root-dir ./ --video-file-list batch.txt
    rm batch.txt

    mv rearpaw.csv ${video_file.baseName}_rearpaw.csv
    mv rearpawsave.csv ${video_file.baseName}_rearpawsave.csv
    """
}
