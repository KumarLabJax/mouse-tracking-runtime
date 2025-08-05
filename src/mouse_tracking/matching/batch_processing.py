"""Memory-efficient batch processing for large video sequences."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mouse_tracking.matching.core import VideoObservations

from mouse_tracking.matching.greedy_matching import vectorized_greedy_matching


class BatchedFrameProcessor:
    """Memory-efficient batch processing for large video sequences.

    This class processes frame sequences in configurable batches to:
    1. Control memory usage for large videos
    2. Enable better cache locality
    3. Allow for future parallel processing of batches
    """

    def __init__(self, batch_size: int = 32):
        """Initialize the batch processor.

        Args:
                batch_size: Number of frames to process together. Larger values use more memory
                                   but may be more efficient. Smaller values use less memory.
        """
        self.batch_size = batch_size

    def process_video_observations(
        self,
        video_observations: "VideoObservations",
        max_cost: float = -np.log(1e-3),
        rotate_pose: bool = False,
    ) -> dict:
        """Process a complete video using batched frame processing.

        Args:
                video_observations: VideoObservations object containing all frame data
                max_cost: Maximum cost threshold for matching
                rotate_pose: Whether to allow 180-degree pose rotation

        Returns:
                Dictionary mapping frame indices to observation matches
        """
        observations = video_observations._observations
        n_frames = len(observations)

        if n_frames <= 1:
            return (
                {0: {i: i for i in range(len(observations[0]))}}
                if n_frames == 1
                else {}
            )

        # Initialize with first frame
        frame_dict = {0: {i: i for i in range(len(observations[0]))}}
        cur_tracklet_id = len(observations[0])

        # Process remaining frames in batches
        for batch_start in range(1, n_frames, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_frames)

            batch_results = self._process_frame_batch(
                video_observations,
                frame_dict,
                cur_tracklet_id,
                batch_start,
                batch_end,
                max_cost,
                rotate_pose,
            )

            frame_dict.update(batch_results["frame_dict"])
            cur_tracklet_id = batch_results["next_tracklet_id"]

        return frame_dict

    def _process_frame_batch(
        self,
        video_observations: "VideoObservations",
        frame_dict: dict,
        cur_tracklet_id: int,
        batch_start: int,
        batch_end: int,
        max_cost: float,
        rotate_pose: bool,
    ) -> dict:
        """Process a single batch of frames.

        Args:
                video_observations: VideoObservations object
                frame_dict: Existing frame matching dictionary
                cur_tracklet_id: Current available tracklet ID
                batch_start: Starting frame index (inclusive)
                batch_end: Ending frame index (exclusive)
                max_cost: Maximum cost threshold
                rotate_pose: Whether to allow pose rotation

        Returns:
                Dictionary with 'frame_dict' and 'next_tracklet_id' keys
        """
        batch_frame_dict = {}
        prev_matches = frame_dict[batch_start - 1]

        # Process each frame in the batch sequentially
        # (Future enhancement could parallelize this within the batch)
        for frame in range(batch_start, batch_end):
            # Calculate cost using vectorized method
            match_costs = video_observations._calculate_costs_vectorized(
                frame - 1, frame, rotate_pose
            )

            # Use optimized greedy matching
            matches = vectorized_greedy_matching(match_costs, max_cost)

            # Map matches to tracklet IDs from previous frame
            tracklet_matches = {}
            for col_idx, row_idx in matches.items():
                tracklet_matches[col_idx] = prev_matches[row_idx]

            # Fill unmatched observations with new tracklet IDs
            for j in range(len(video_observations._observations[frame])):
                if j not in tracklet_matches:
                    tracklet_matches[j] = cur_tracklet_id
                    cur_tracklet_id += 1

            batch_frame_dict[frame] = tracklet_matches
            prev_matches = tracklet_matches

        return {"frame_dict": batch_frame_dict, "next_tracklet_id": cur_tracklet_id}
