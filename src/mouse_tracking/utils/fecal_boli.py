"""Utilities for fecal boli functionality."""

import glob

import h5py
import imageio
import numpy as np
import pandas as pd

from mouse_tracking.utils.rendering import plot_frame_info
from mouse_tracking.utils.static_objects import plot_keypoints
from mouse_tracking.utils.timers import print_time


def aggregate_folder_data(folder: str, depth: int = 2, num_bins: int = -1):
    """Aggregates fecal boli data in a folder into a table.

    Args:
            folder: project folder
            depth: expected subfolder depth
            num_bins: number of bins to read in (value < 0 reads all)

    Returns:
            pd.DataFrame containing the fecal boli counts over time

    Notes:
            Open field project folder looks like [computer]/[date]/[video]_pose_est_v6.h5 files
            depth defaults to have these 2 folders

    Todo:
            Currently this makes some bad assumptions about data.
                    Time is assumed to be 1-minute intervals. Another field stores the times when they occur
                    _pose_est_v6 is searched, but this is currently a proposed v7 feature
                    no error handling is present...
    """
    pose_files = glob.glob(folder + "/" + "*/" * depth + "*_pose_est_v6.h5")

    max_bin_count = None if num_bins < 0 else num_bins

    read_data = []
    for cur_file in pose_files:
        with h5py.File(cur_file, "r") as f:
            counts = f["dynamic_objects/fecal_boli/counts"][:].flatten().astype(float)
            # Clip the number of bins if requested
            if max_bin_count is not None:
                if len(counts) > max_bin_count:
                    counts = counts[:max_bin_count]
                elif len(counts) < max_bin_count:
                    counts = np.pad(
                        counts,
                        (0, max_bin_count - len(counts)),
                        "constant",
                        constant_values=np.nan,
                    )
        new_df = pd.DataFrame(counts, columns=["count"])
        new_df["minute"] = np.arange(len(new_df))
        new_df["NetworkFilename"] = cur_file[len(folder) : len(cur_file) - 15] + ".avi"
        pivot = new_df.pivot(index="NetworkFilename", columns="minute", values="count")
        read_data.append(pivot)

    all_data = pd.concat(read_data).reset_index(drop=False)
    return all_data


def render_fecal_boli_video(in_video: str, in_pose: str, out_video: str):
    """
    Renders fecal boli on a frame.

    Args:
            in_video: The input video file
            in_pose: The input pose file
            out_video: The output video file
    """
    # Open the input video
    vid_writer = imageio.get_writer(out_video, fps=1)

    # Load the pose data
    with h5py.File(in_pose, "r") as f:
        fecal_boli = f["dynamic_objects/fecal_boli/points"][...]
        fecal_boli_counts = f["dynamic_objects/fecal_boli/counts"][...]
        fecal_boli_frames = f["dynamic_objects/fecal_boli/sample_indices"][...]

    video_reader = imageio.get_reader(in_video)
    video_done = False
    prediction_idx = 0

    while not video_done:
        try:
            prediction_frame = fecal_boli_frames[prediction_idx]
            input_frame = video_reader.get_data(prediction_frame)
        except StopIteration:
            video_done = True
            break

        fecal_boli_count_in_frame = fecal_boli_counts[prediction_idx]
        fecal_boli_data = fecal_boli[prediction_idx, : int(fecal_boli_count_in_frame)]
        if fecal_boli_count_in_frame > 0:
            rendered_frame = plot_keypoints(
                fecal_boli_data, input_frame, is_yx=True, radius=5, alpha=0.5
            )
        else:
            rendered_frame = input_frame

        rendered_frame = plot_frame_info(
            rendered_frame, f"Video Timestamp: {print_time(prediction_frame)}"
        )

        # Write the frame to the output video
        vid_writer.append_data(rendered_frame)
        prediction_idx += 1
        if prediction_idx >= len(fecal_boli_frames):
            video_done = True
            break

    vid_writer.close()
