"""Pose file inspection utilities."""

import re
from pathlib import Path

import h5py
import numpy as np

from mouse_tracking.core.config.pose_utils import PoseUtilsConfig
from mouse_tracking.utils.arrays import safe_find_first
from mouse_tracking.utils.hashing import hash_file

CONFIG = PoseUtilsConfig()


def inspect_pose_v2(pose_file, pad: int = 150, duration: int = 108000) -> dict:
    """Inspects a single mouse pose file v2 for coverage metrics.

    Args:
        pose_file: The pose file to inspect
        pad: pad size expected in the beginning
        duration: expected duration of experiment

    Returns:
        Dict containing the following keyed data:
            first_frame_pose: First frame where the pose data appeared
            first_frame_full_high_conf: First frame with 12 keypoints at high confidence
            pose_counts: total number of poses predicted
            missing_poses: missing poses in the primary duration of the video
            missing_keypoint_frames: number of frames which don't contain 12 keypoints in the primary duration
    """
    with h5py.File(pose_file, "r") as f:
        pose_version = f["poseest"].attrs["version"][0]
        if pose_version != 2:
            msg = f"Only v2 pose files are supported for inspection. {pose_file} is version {pose_version}"
            raise ValueError(msg)
        pose_quality = f["poseest/confidence"][:]

    num_keypoints = np.sum(pose_quality > CONFIG.MIN_JABS_CONFIDENCE, axis=1)
    high_conf_keypoints = np.all(
        pose_quality > CONFIG.MIN_HIGH_CONFIDENCE, axis=2
    ).squeeze(1)

    return {
        "first_frame_pose": safe_find_first(high_conf_keypoints),
        "first_frame_full_high_conf": safe_find_first(high_conf_keypoints),
        "pose_counts": np.sum(num_keypoints > CONFIG.MIN_JABS_CONFIDENCE),
        "missing_poses": duration
        - np.sum((num_keypoints > CONFIG.MIN_JABS_CONFIDENCE)[pad : pad + duration]),
        "missing_keypoint_frames": np.sum(num_keypoints[pad : pad + duration] != 12),
    }


def inspect_pose_v6(pose_file, pad: int = 150, duration: int = 108000) -> dict:
    """Inspects a single mouse pose file v6 for coverage metrics.

    Args:
        pose_file: The pose file to inspect
        pad: duration of data skipped in the beginning (not observation period)
        duration: observation duration of experiment

    Returns:
        Dict containing the following keyed data:
            pose_file: The pose file inspected
            pose_hash: The blake2b hash of the pose file
            video_name: The video name associated with the pose file (no extension)
            video_duration: Duration of the video
            corners_present: If the corners are present in the pose file
            first_frame_pose: First frame where the pose data appeared
            first_frame_full_high_conf: First frame with 12 keypoints > 0.75 confidence
            first_frame_jabs: First frame with 3 keypoints > 0.3 confidence
            first_frame_gait: First frame > 0.3 confidence for base tail and rear paws keypoints
            first_frame_seg: First frame where segmentation data was assigned an id
            pose_counts: Total number of poses predicted
            seg_counts: Total number of segmentations matched with poses
            missing_poses: Missing poses in the observation duration of the video
            missing_segs: Missing segmentations in the observation duration of the video
            pose_tracklets: Number of tracklets in the observation duration
            missing_keypoint_frames: Number of frames which don't contain 12 keypoints in the observation duration
    """
    with h5py.File(pose_file, "r") as f:
        pose_version = f["poseest"].attrs["version"][0]
        if pose_version < 6:
            msg = f"Only v6+ pose files are supported for inspection. {pose_file} is version {pose_version}"
            raise ValueError(msg)
        pose_counts = f["poseest/instance_count"][:]
        if np.max(pose_counts) > 1:
            msg = f"Only single mouse pose files are supported for inspection. {pose_file} contains multiple instances"
            raise ValueError(msg)
        pose_quality = f["poseest/confidence"][:]
        pose_tracks = f["poseest/instance_track_id"][:]
        seg_ids = f["poseest/longterm_seg_id"][:]
        corners_present = "static_objects/corners" in f

    num_keypoints = 12 - np.sum(pose_quality.squeeze(1) == 0, axis=1)

    # Keep 2 folders if present for video name
    folder_name = "/".join(Path(pose_file).parts[-3:-1]) + "/"

    high_conf_keypoints = np.all(
        pose_quality > CONFIG.MIN_HIGH_CONFIDENCE, axis=2
    ).squeeze(1)

    jabs_keypoints = np.sum(pose_quality > CONFIG.MIN_JABS_CONFIDENCE, axis=2).squeeze(
        1
    )

    gait_keypoints = np.all(
        pose_quality[
            :,
            :,
            [
                CONFIG.BASE_TAIL_INDEX,
                CONFIG.LEFT_REAR_PAW_INDEX,
                CONFIG.RIGHT_REAR_PAW_INDEX,
            ],
        ]
        > CONFIG.MIN_GAIT_CONFIDENCE,
        axis=2,
    ).squeeze(1)

    return {
        "pose_file": Path(pose_file).name,
        "pose_hash": hash_file(Path(pose_file)),
        "video_name": folder_name
        + re.sub("_pose_est_v[0-9]+", "", Path(pose_file).stem),
        "video_duration": pose_counts.shape[0],
        "corners_present": corners_present,
        "first_frame_pose": safe_find_first(pose_counts > 0),
        "first_frame_full_high_conf": safe_find_first(high_conf_keypoints),
        "first_frame_jabs": safe_find_first(
            jabs_keypoints >= CONFIG.MIN_JABS_KEYPOINTS
        ),
        "first_frame_gait": safe_find_first(gait_keypoints),
        "first_frame_seg": safe_find_first(seg_ids > 0),
        "pose_counts": np.sum(pose_counts),
        "seg_counts": np.sum(seg_ids > 0),
        "missing_poses": duration - np.sum(pose_counts[pad : pad + duration]),
        "missing_segs": duration - np.sum(seg_ids[pad : pad + duration] > 0),
        "pose_tracklets": len(
            np.unique(
                pose_tracks[pad : pad + duration][
                    pose_counts[pad : pad + duration] == 1
                ]
            )
        ),
        "missing_keypoint_frames": np.sum(num_keypoints[pad : pad + duration] != 12),
    }
