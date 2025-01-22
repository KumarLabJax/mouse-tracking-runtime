manifest {
    name = "Kumar Lab Dynamic Object Detection"
}

process fecal_boli {
    label: "gpu"
    label: "tracking"
    
    input:
    file video_file
    val out_pose_filename

    output:
    file out_pose_file

    script:
    """
    python3 $code_dir/infer_fecal_boli.py --video $video_file --out-file $out_pose_filename
    """
}
