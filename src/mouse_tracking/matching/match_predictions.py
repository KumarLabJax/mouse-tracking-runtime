"""Stitch tracklets within a pose file."""

import time

import h5py
import numpy as np

from mouse_tracking.matching import VideoObservations
from mouse_tracking.utils.timers import time_accumulator
from mouse_tracking.utils.writers import (
    write_pose_v3_data,
    write_pose_v4_data,
    write_v6_tracklets,
)


def match_predictions(pose_file):
    """Reads in pose and segmentation data to match data over the time dimension.

    Args:
            pose_file: pose file to modify in-place

    Notes:
            This function only applies the optimal settings from identity repository.
    """
    performance_accumulator = time_accumulator(
        3, ["Matching Poses", "Tracklet Generation", "Tracklet Stitching"]
    )
    t1 = time.time()
    video_observations = VideoObservations.from_pose_file(pose_file, 0.0)
    t2 = time.time()
    # video_observations.generate_greedy_tracklets(rotate_pose=True, num_threads=1)
    video_observations.generate_greedy_tracklets_vectorized(rotate_pose=True)
    with h5py.File(pose_file, "r") as f:
        pose_shape = f["poseest/points"].shape[:2]
        seg_shape = f["poseest/seg_data"].shape[:2]
    new_pose_ids, new_seg_ids = video_observations.get_id_mat(pose_shape, seg_shape)

    # Stitch the tracklets together
    t3 = time.time()
    video_observations.stitch_greedy_tracklets_optimized(
        num_tracks=None, prioritize_long=True
    )
    translated_tracks = video_observations.stitch_translation
    stitched_pose = np.vectorize(lambda x: translated_tracks.get(x, 0))(new_pose_ids)
    stitched_seg = np.vectorize(lambda x: translated_tracks.get(x, 0))(new_seg_ids)
    centers = video_observations.get_embed_centers()
    t4 = time.time()
    performance_accumulator.add_batch_times([t1, t2, t3, t4])

    # Write data out
    # We need to overwrite original tracklet data
    write_pose_v3_data(pose_file, instance_track=new_pose_ids)
    # Also overwrite stitched tracklet data
    mask = stitched_pose == 0
    write_pose_v4_data(pose_file, mask, stitched_pose, centers)
    # Finally, overwrite segmentation data
    write_v6_tracklets(pose_file, new_seg_ids, stitched_seg)
    performance_accumulator.print_performance()
