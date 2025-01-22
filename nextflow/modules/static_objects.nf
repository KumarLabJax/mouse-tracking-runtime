manifest {
    name = "Kumar Lab Static Object Detection"
}

process arena_corners {
    label: "gpu"
    label: "tracking"
    
    input:
    file video_file
    val out_pose_filename

    output:
    file out_pose_file

    script:
    """
    python3 $code_dir/infer_arena_corner.py --video $video_file --out-file $out_pose_filename
    """
}

process food_hopper {
    label: "gpu"
    label: "tracking"
    
    input:
    file video_file
    val out_pose_filename

    output:
    file out_pose_file

    script:
    """
    python3 $code_dir/infer_food_hopper.py --video $video_file --out-file $out_pose_filename
    """
}

process lixit {
    label: "gpu"
    label: "tracking"
    
    input:
    file video_file
    val out_pose_filename

    output:
    file out_pose_file

    script:
    """
    python3 $code_dir/infer_lixit.py --video $video_file --out-file $out_pose_filename
    """
}
