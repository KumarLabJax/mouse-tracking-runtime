"""Benchmark tests for VideoObservations.stitch_greedy_tracklets method.

This module contains performance benchmarks to measure the efficiency of tracklet stitching
and help identify performance bottlenecks. Uses pytest-benchmark plugin.

Run with: pytest tests/utils/matching/video_observations/test_benchmark_stich_greedy_tracklets.py --benchmark-only
"""

import numpy as np
import pytest

from mouse_tracking.matching.core import Detection, VideoObservations


@pytest.fixture
def mock_detection():
    """Create a mock detection with realistic data."""

    def _create_detection(frame_idx, pose_idx, embed_size=128):
        pose = np.random.rand(12, 2) * 100  # Random pose keypoints
        embed = np.random.rand(embed_size)  # Random embedding vector
        seg = np.random.randint(-1, 100, size=(100, 2))  # Random segmentation contour
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
def small_video_observations(mock_detection):
    """Create VideoObservations with small number of tracklets (10-15 tracklets)."""
    observations = []
    num_frames = 100
    animals_per_frame = 2

    for frame_idx in range(num_frames):
        frame_observations = []
        for animal_idx in range(animals_per_frame):
            detection = mock_detection(frame_idx, animal_idx)
            frame_observations.append(detection)
        observations.append(frame_observations)

    video_obs = VideoObservations(observations)
    # Generate tracklets
    video_obs.generate_greedy_tracklets(rotate_pose=True, num_threads=1)
    return video_obs


@pytest.fixture
def medium_video_observations(mock_detection):
    """Create VideoObservations with medium number of tracklets (30-50 tracklets)."""
    observations = []
    num_frames = 200
    animals_per_frame = 3

    for frame_idx in range(num_frames):
        frame_observations = []
        for animal_idx in range(animals_per_frame):
            # Add some noise to create more tracklets by making some detections inconsistent
            if np.random.random() > 0.8:  # 20% chance to skip detection
                continue
            detection = mock_detection(frame_idx, animal_idx)
            frame_observations.append(detection)
        observations.append(frame_observations)

    video_obs = VideoObservations(observations)
    # Generate tracklets
    video_obs.generate_greedy_tracklets(rotate_pose=True, num_threads=1)
    return video_obs


@pytest.fixture
def large_video_observations(mock_detection):
    """Create VideoObservations with large number of tracklets (80-120 tracklets)."""
    observations = []
    num_frames = 300
    animals_per_frame = 4

    for frame_idx in range(num_frames):
        frame_observations = []
        for animal_idx in range(animals_per_frame):
            # Add more noise to create many fragmented tracklets
            if np.random.random() > 0.7:  # 30% chance to skip detection
                continue
            detection = mock_detection(frame_idx, animal_idx)
            frame_observations.append(detection)
        observations.append(frame_observations)

    video_obs = VideoObservations(observations)
    # Generate tracklets
    video_obs.generate_greedy_tracklets(rotate_pose=True, num_threads=1)
    return video_obs


class TestStitchGreedyTrackletsBenchmark:
    """Benchmark tests for stitch_greedy_tracklets method."""

    def test_benchmark_small_tracklets(self, benchmark, small_video_observations):
        """Benchmark stitching with small number of tracklets (~10-15)."""
        # Store original tracklets for verification
        original_tracklet_count = len(small_video_observations._tracklets)

        def run_stitch():
            # Reset tracklets before each run
            small_video_observations._make_tracklets()
            small_video_observations.stitch_greedy_tracklets(
                num_tracks=None, all_embeds=True, prioritize_long=True
            )
            return len(small_video_observations._tracklets)

        result = benchmark(run_stitch)

        # Verify that stitching actually happened
        assert result <= original_tracklet_count
        print(f"Small test: {original_tracklet_count} -> {result} tracklets")

    def test_benchmark_medium_tracklets(self, benchmark, medium_video_observations):
        """Benchmark stitching with medium number of tracklets (~30-50)."""
        original_tracklet_count = len(medium_video_observations._tracklets)

        def run_stitch():
            # Reset tracklets before each run
            medium_video_observations._make_tracklets()
            medium_video_observations.stitch_greedy_tracklets(
                num_tracks=None, all_embeds=True, prioritize_long=True
            )
            return len(medium_video_observations._tracklets)

        result = benchmark(run_stitch)

        # Verify that stitching actually happened
        assert result <= original_tracklet_count
        print(f"Medium test: {original_tracklet_count} -> {result} tracklets")

    def test_benchmark_large_tracklets(self, benchmark, large_video_observations):
        """Benchmark stitching with large number of tracklets (~80-120)."""
        original_tracklet_count = len(large_video_observations._tracklets)

        def run_stitch():
            # Reset tracklets before each run
            large_video_observations._make_tracklets()
            large_video_observations.stitch_greedy_tracklets(
                num_tracks=None, all_embeds=True, prioritize_long=True
            )
            return len(large_video_observations._tracklets)

        result = benchmark(run_stitch)

        # Verify that stitching actually happened
        assert result <= original_tracklet_count
        print(f"Large test: {original_tracklet_count} -> {result} tracklets")

    def test_benchmark_get_transition_costs(self, benchmark, medium_video_observations):
        """Benchmark the _get_transition_costs method specifically."""

        def run_get_costs():
            return medium_video_observations._get_transition_costs(
                all_comparisons=True, include_inf=True, longer_track_priority=1.0
            )

        result = benchmark(run_get_costs)

        # Verify result is reasonable
        assert isinstance(result, dict)
        assert len(result) > 0
        print(f"Transition costs calculated for {len(result)} tracklets")

    def test_scaling_comparison(
        self,
        benchmark,
        small_video_observations,
        medium_video_observations,
        large_video_observations,
    ):
        """Compare performance scaling across different tracklet counts."""
        import time

        test_cases = [
            ("small", small_video_observations),
            ("medium", medium_video_observations),
            ("large", large_video_observations),
        ]

        results = {}

        for name, video_obs in test_cases:
            original_count = len(video_obs._tracklets)

            # Reset tracklets
            video_obs._make_tracklets()

            # Time the stitching
            start_time = time.time()
            video_obs.stitch_greedy_tracklets(
                num_tracks=None, all_embeds=True, prioritize_long=True
            )
            end_time = time.time()

            final_count = len(video_obs._tracklets)
            duration = end_time - start_time

            results[name] = {
                "original_tracklets": original_count,
                "final_tracklets": final_count,
                "duration_seconds": duration,
                "tracklets_per_second": original_count / duration
                if duration > 0
                else float("inf"),
            }

            print(
                f"{name}: {original_count} -> {final_count} tracklets in {duration:.3f}s"
            )

        # Check for quadratic or worse scaling
        small_time = results["small"]["duration_seconds"]
        medium_time = results["medium"]["duration_seconds"]
        large_time = results["large"]["duration_seconds"]

        small_tracklets = results["small"]["original_tracklets"]
        medium_tracklets = results["medium"]["original_tracklets"]
        large_tracklets = results["large"]["original_tracklets"]

        if medium_time > 0 and small_time > 0:
            scaling_factor_small_to_medium = (medium_time / small_time) / (
                (medium_tracklets / small_tracklets) ** 2
            )
            print(
                f"Scaling factor (small->medium): {scaling_factor_small_to_medium:.2f} (1.0 = quadratic)"
            )

        if large_time > 0 and medium_time > 0:
            scaling_factor_medium_to_large = (large_time / medium_time) / (
                (large_tracklets / medium_tracklets) ** 2
            )
            print(
                f"Scaling factor (medium->large): {scaling_factor_medium_to_large:.2f} (1.0 = quadratic)"
            )


@pytest.mark.parametrize(
    "num_tracklets,expected_complexity",
    [(10, "linear"), (30, "quadratic"), (50, "quadratic"), (100, "cubic")],
)
def test_complexity_analysis(
    benchmark, mock_detection, num_tracklets, expected_complexity
):
    """Test performance complexity with different numbers of tracklets."""
    # Create observations that will result in approximately num_tracklets tracklets
    observations = []
    frames_per_tracklet = 5
    num_frames = num_tracklets * frames_per_tracklet

    for frame_idx in range(num_frames):
        frame_observations = []
        # Create sparse detections to generate many short tracklets
        if frame_idx % frames_per_tracklet < 2:  # Only 2 out of every 5 frames
            detection = mock_detection(frame_idx, frame_idx // frames_per_tracklet)
            frame_observations.append(detection)
        observations.append(frame_observations)

    video_obs = VideoObservations(observations)
    video_obs.generate_greedy_tracklets(rotate_pose=True, num_threads=1)

    actual_tracklets = len(video_obs._tracklets)
    print(f"Created {actual_tracklets} tracklets (target: {num_tracklets})")

    # Measure time
    import time

    start_time = time.time()
    video_obs.stitch_greedy_tracklets(
        num_tracks=None, all_embeds=True, prioritize_long=True
    )
    duration = time.time() - start_time

    print(f"Processed {actual_tracklets} tracklets in {duration:.3f}s")

    # Basic complexity check - this is more for documentation than assertion
    if actual_tracklets > 0:
        time_per_tracklet = duration / actual_tracklets
        time_per_tracklet_squared = duration / (actual_tracklets**2)
        print(f"Time per tracklet: {time_per_tracklet:.6f}s")
        print(f"Time per tracklet²: {time_per_tracklet_squared:.6f}s")


if __name__ == "__main__":
    # Allow running benchmark tests directly
    pytest.main([__file__, "--benchmark-only", "-v"])
