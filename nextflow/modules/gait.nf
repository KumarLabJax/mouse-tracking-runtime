process GENERATE_GAIT_H5 {
    label "gait"
    label "cpu"
    label "r_gait_h5"
    
    input:
    tuple path(video_file) , path(pose_file)

    output:
    path("${video_file.baseName}_gait.h5"), emit: gait_file

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
    python -u ${params.gait_code_dir}/gengaitstats.py --batch-file batch.txt --root-dir ./ --out-file ${video_file.baseName}_gait.h5
    rm batch.txt
    """
}


process GENERATE_GAIT_BIN {
    label "gait"
    label "cpu"
    label "r_gait_bin"

    input:
    tuple path(gait_file), val(speed_bin)

    output:
    path("${gait_file.baseName}_gait_out_${speed_bin}cm.csv"), emit: gait_bin_csv

    script:
    """
    python ${params.gait_code_dir}/summarizegaitcsv.py --gait-h5 ${gait_file} --speed ${speed_bin} > ${gait_file.baseName}_gait_out_${speed_bin}cm.csv
    """
}