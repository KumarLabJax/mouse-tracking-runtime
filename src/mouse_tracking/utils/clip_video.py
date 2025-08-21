"""Produce a clip of pose and video data based on when a mouse is first detected."""

import subprocess
from pathlib import Path

import numpy as np

from mouse_tracking.utils import writers
from mouse_tracking.utils.pose import find_first_pose_file
from mouse_tracking.utils.timers import print_time


def clip_video(in_video, in_pose, out_video, out_pose, frame_start, frame_end):
    """Clips a video and pose file.

    Args:
            in_video: path indicating the video to copy frames from
            in_pose: path indicating the pose file to copy frames from
            out_video: path indicating the output video
            out_pose: path indicating the output pose file
            frame_start: first frame in the video to copy
            frame_end: last frame in the video to copy

    Notes:
            This function requires ffmpeg to be installed on the system.
    """
    if not Path(in_video).exists():
        msg = f"{in_video} does not exist"
        raise FileNotFoundError(msg)
    if not Path(in_pose).exists():
        msg = f"{in_pose} does not exist"
        raise FileNotFoundError(msg)
    if not isinstance(frame_start, int | np.integer):
        msg = f"frame_start must be an integer, not {type(frame_start)}"
        raise TypeError(msg)
    if not isinstance(frame_end, int | np.integer):
        msg = f"frame_start must be an integer, not {type(frame_end)}"
        raise TypeError(msg)

    ffmpeg_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "panic",
        "-r",
        "30",
        "-i",
        in_video,
        "-an",
        "-sn",
        "-dn",
        "-vf",
        f"select=gte(n\,{frame_start}),setpts=PTS-STARTPTS",
        "-vframes",
        f"{frame_end - frame_start}",
        "-f",
        "mp4",
        "-c:v",
        "libx264",
        "-preset",
        "veryslow",
        "-profile:v",
        "main",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "30",
        "-y",
        out_video,
    ]

    subprocess.run(ffmpeg_command, check=False)

    writers.write_pose_clip(in_pose, out_pose, range(frame_start, frame_end))


def clip_video_auto(
    in_video: str,
    in_pose: str,
    out_video: str,
    out_pose: str,
    frame_offset: int = 150,  # Default 5 seconds in frames
    observation_duration: int = 30 * 60 * 60,  # Default 1 hour in frames
    confidence_threshold: float = 0.5,  # Default confidence threshold
    num_keypoints: int = 12,  # Default number of keypoints
):
    """Clip a video and pose file based on the first detected pose."""
    first_frame = find_first_pose_file(in_pose, confidence_threshold, num_keypoints)
    output_start_frame = np.maximum(first_frame - frame_offset, 0)
    output_end_frame = output_start_frame + frame_offset + observation_duration
    print(
        f"Clipping video from frames {output_start_frame} ({print_time(output_start_frame)}) to {output_end_frame} ({print_time(output_end_frame)})"
    )
    clip_video(
        in_video, in_pose, out_video, out_pose, output_start_frame, output_end_frame
    )


def clip_video_manual(
    in_video: str,
    in_pose: str,
    out_video: str,
    out_pose: str,
    frame_start: int,
    observation_duration: int = 30 * 60 * 60,  # Default 1 hour in frames
):
    """Clip a video and pose file based on a manually specified start frame."""
    first_frame = np.maximum(frame_start, 0)
    output_end_frame = first_frame + observation_duration
    print(
        f"Clipping video from frames {first_frame} ({print_time(first_frame)}) to {output_end_frame} ({print_time(output_end_frame)})"
    )
    clip_video(in_video, in_pose, out_video, out_pose, first_frame, output_end_frame)
