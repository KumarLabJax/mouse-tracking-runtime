"""Shared fixtures for VideoObservations testing.

This module provides shared test fixtures and utilities for testing the VideoObservations
class and its methods, particularly the stitch_greedy_tracklets functionality.
"""

import numpy as np
import pytest

from mouse_tracking.matching.core import Detection, Tracklet, VideoObservations


@pytest.fixture
def basic_detection():
    """Create a function that generates basic Detection objects with configurable parameters."""

    def _create_detection(
        frame_idx: int = 0,
        pose_idx: int = 0,
        embed_size: int = 128,
        pose_shape: tuple = (12, 2),
        seg_shape: tuple = (100, 2),
        embed_value: float | None = None,
        pose_coords: tuple | None = None,
    ):
        """Create a Detection with specified parameters.

        Args:
            frame_idx: Frame index for the detection
            pose_idx: Pose index within the frame
            embed_size: Size of the embedding vector
            pose_shape: Shape of pose data
            seg_shape: Shape of segmentation data
            embed_value: Fixed value for embedding (random if None)
            pose_coords: Fixed coordinates for pose center (random if None)

        Returns:
            Detection object with specified parameters
        """
        # Create pose data
        if pose_coords is not None:
            pose = np.zeros(pose_shape, dtype=np.float32)
            center_x, center_y = pose_coords
            # Create pose keypoints around the center
            for i in range(pose_shape[0]):
                pose[i] = [
                    center_x + np.random.uniform(-10, 10),
                    center_y + np.random.uniform(-10, 10),
                ]
        else:
            pose = np.random.rand(*pose_shape) * 100

        # Create embedding
        if embed_value is not None:
            embed = np.full(embed_size, embed_value, dtype=np.float32)
        else:
            embed = np.random.rand(embed_size).astype(np.float32)

        # Create segmentation data
        seg = np.random.randint(-1, 100, size=seg_shape, dtype=np.int32)

        return Detection(
            frame=frame_idx,
            pose_idx=pose_idx,
            pose=pose,
            embed=embed,
            seg_idx=pose_idx,
            seg=seg,
        )

    return _create_detection


@pytest.fixture
def simple_tracklet(basic_detection):
    """Create a simple tracklet with a few detections."""

    def _create_tracklet(
        track_id: int = 1,
        frame_range: tuple = (0, 5),
        pose_coords: tuple = (50, 50),
        embed_value: float = 0.5,
    ):
        """Create a tracklet with detections across specified frames.

        Args:
            track_id: ID for the tracklet
            frame_range: (start_frame, end_frame) for the tracklet
            pose_coords: Center coordinates for poses
            embed_value: Fixed embedding value for all detections

        Returns:
            Tracklet object
        """
        detections = []
        for frame in range(frame_range[0], frame_range[1]):
            detection = basic_detection(
                frame_idx=frame,
                pose_idx=0,
                embed_value=embed_value,
                pose_coords=pose_coords,
            )
            detections.append(detection)

        return Tracklet(track_id, detections)

    return _create_tracklet


@pytest.fixture
def minimal_video_observations(basic_detection):
    """Create VideoObservations with minimal data (2 tracklets)."""
    observations = []

    # Create two simple tracklets
    # Tracklet 1: frames 0-4
    for frame in range(5):
        detection = basic_detection(
            frame_idx=frame,
            pose_idx=0,
            embed_value=0.1,
            pose_coords=(20, 20),
        )
        observations.append([detection])

    # Gap (no detections)
    for _ in range(5, 10):
        observations.append([])

    # Tracklet 2: frames 10-14
    for frame in range(10, 15):
        detection = basic_detection(
            frame_idx=frame,
            pose_idx=0,
            embed_value=0.9,
            pose_coords=(80, 80),
        )
        observations.append([detection])

    video_obs = VideoObservations(observations)
    video_obs.generate_greedy_tracklets(rotate_pose=False, num_threads=1)
    return video_obs


@pytest.fixture
def fragmented_video_observations(basic_detection):
    """Create VideoObservations with many small tracklets that can be stitched."""
    observations = []

    # Create several small tracklets with similar embeddings that should be stitched
    tracklet_configs = [
        # (start_frame, duration, embed_value, pose_coords)
        (0, 3, 0.1, (10, 10)),  # Tracklet 1
        (5, 2, 0.11, (10, 10)),  # Similar to tracklet 1, should stitch
        (10, 4, 0.2, (50, 50)),  # Tracklet 2
        (16, 3, 0.21, (50, 50)),  # Similar to tracklet 2, should stitch
        (25, 2, 0.3, (90, 90)),  # Tracklet 3
        (30, 3, 0.31, (90, 90)),  # Similar to tracklet 3, should stitch
    ]

    # Initialize all frames as empty
    total_frames = 35
    for _ in range(total_frames):
        observations.append([])

    # Add detections according to tracklet configs
    for start_frame, duration, embed_value, pose_coords in tracklet_configs:
        for offset in range(duration):
            frame = start_frame + offset
            detection = basic_detection(
                frame_idx=frame,
                pose_idx=0,
                embed_value=embed_value,
                pose_coords=pose_coords,
            )
            observations[frame] = [detection]

    video_obs = VideoObservations(observations)
    video_obs.generate_greedy_tracklets(rotate_pose=False, num_threads=1)
    return video_obs


@pytest.fixture
def single_tracklet_video_observations(basic_detection):
    """Create VideoObservations with only one tracklet (edge case)."""
    observations = []

    # Single tracklet: frames 0-9
    for frame in range(10):
        detection = basic_detection(
            frame_idx=frame,
            pose_idx=0,
            embed_value=0.5,
            pose_coords=(50, 50),
        )
        observations.append([detection])

    video_obs = VideoObservations(observations)
    video_obs.generate_greedy_tracklets(rotate_pose=False, num_threads=1)
    return video_obs


@pytest.fixture
def empty_video_observations():
    """Create VideoObservations with no tracklets (edge case)."""
    observations = []

    # Create empty frames
    for _ in range(10):
        observations.append([])

    video_obs = VideoObservations(observations)
    # Don't call generate_greedy_tracklets for empty data - it will fail
    # Instead, manually set up the minimal state
    video_obs._tracklets = []
    video_obs._tracklet_gen_method = None
    return video_obs


@pytest.fixture
def complex_video_observations(basic_detection):
    """Create VideoObservations with complex stitching scenarios."""
    observations = []
    total_frames = 100

    # Initialize all frames as empty
    for _ in range(total_frames):
        observations.append([])

    # Create complex tracklet patterns
    tracklet_patterns = [
        # Long tracklets that should remain separate
        (0, 20, 0.1, (10, 10)),  # Long tracklet 1
        (25, 25, 0.9, (90, 90)),  # Long tracklet 2 (different embedding)
        # Short tracklets that should stitch together
        (55, 3, 0.2, (30, 30)),  # Part 1 of animal
        (60, 4, 0.21, (30, 30)),  # Part 2 of same animal
        (67, 2, 0.19, (30, 30)),  # Part 3 of same animal
        # Overlapping tracklets (should not stitch)
        (75, 10, 0.3, (60, 60)),  # Overlapping tracklet 1
        (80, 10, 0.31, (60, 60)),  # Overlapping tracklet 2 (slight overlap)
        # Very short tracklets
        (92, 1, 0.4, (70, 70)),  # Single frame
        (95, 2, 0.41, (70, 70)),  # Two frames
    ]

    # Add detections according to patterns
    for start_frame, duration, embed_value, pose_coords in tracklet_patterns:
        for offset in range(duration):
            frame = start_frame + offset
            if frame < total_frames:
                detection = basic_detection(
                    frame_idx=frame,
                    pose_idx=0,
                    embed_value=embed_value,
                    pose_coords=pose_coords,
                )
                observations[frame] = [detection]

    video_obs = VideoObservations(observations)
    video_obs.generate_greedy_tracklets(rotate_pose=False, num_threads=1)
    return video_obs


@pytest.fixture
def tracklet_lengths_fixture():
    """Return function to calculate tracklet lengths."""

    def _get_tracklet_lengths(video_observations):
        """Get lengths of all tracklets in VideoObservations."""
        return [len(tracklet.frames) for tracklet in video_observations._tracklets]

    return _get_tracklet_lengths


@pytest.fixture
def tracklet_ids_fixture():
    """Return function to extract tracklet IDs."""

    def _get_tracklet_ids(video_observations):
        """Get all tracklet IDs from VideoObservations."""
        return [tracklet.track_id for tracklet in video_observations._tracklets]

    return _get_tracklet_ids


@pytest.fixture
def verify_no_overlaps_fixture():
    """Return function to verify tracklets don't overlap."""

    def _verify_no_overlaps(video_observations):
        """Verify that no tracklets overlap in frames."""
        tracklets = video_observations._tracklets
        for i, tracklet_1 in enumerate(tracklets):
            for j, tracklet_2 in enumerate(tracklets[i + 1 :], i + 1):
                assert not tracklet_1.overlaps_with(tracklet_2), (
                    f"Tracklet {i} overlaps with tracklet {j}"
                )

    return _verify_no_overlaps


@pytest.fixture
def stitching_verification_fixture():
    """Return function to verify stitching results are valid."""

    def _verify_stitching_results(
        original_tracklets, stitched_tracklets, original_count, final_count
    ):
        """Verify that stitching results are valid.

        Args:
            original_tracklets: List of tracklets before stitching
            stitched_tracklets: List of tracklets after stitching
            original_count: Original number of tracklets
            final_count: Final number of tracklets after stitching

        Returns:
            dict with verification results
        """
        # Basic count check
        assert len(stitched_tracklets) == final_count, (
            f"Expected {final_count} tracklets, got {len(stitched_tracklets)}"
        )

        # Should have fewer or same number of tracklets
        assert final_count <= original_count, (
            "Stitching should not increase tracklet count"
        )

        # All frames should still be covered
        original_frames = set()
        for tracklet in original_tracklets:
            original_frames.update(tracklet.frames)

        stitched_frames = set()
        for tracklet in stitched_tracklets:
            stitched_frames.update(tracklet.frames)

        assert original_frames == stitched_frames, (
            "Frame coverage should not change after stitching"
        )

        # No overlaps should exist
        for i, tracklet_1 in enumerate(stitched_tracklets):
            for j, tracklet_2 in enumerate(stitched_tracklets[i + 1 :], i + 1):
                assert not tracklet_1.overlaps_with(tracklet_2), (
                    f"Stitched tracklet {i} overlaps with tracklet {j}"
                )

        return {
            "original_count": original_count,
            "final_count": final_count,
            "reduction": original_count - final_count,
            "reduction_percentage": (original_count - final_count)
            / original_count
            * 100
            if original_count > 0
            else 0,
        }

    return _verify_stitching_results
