"""Tests for BatchedFrameProcessor.process_video_observations method."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from mouse_tracking.matching.batch_processing import BatchedFrameProcessor


class TestProcessVideoObservations:
    """Test process_video_observations method."""

    def test_process_video_observations_basic(self):
        """Test basic video processing functionality."""
        processor = BatchedFrameProcessor(batch_size=2)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock(), Mock()],  # Frame 0: 2 detections
            [Mock(), Mock()],  # Frame 1: 2 detections
            [Mock()],  # Frame 2: 1 detection
        ]

        # Mock the _process_frame_batch method
        with patch.object(processor, "_process_frame_batch") as mock_batch_process:
            mock_batch_process.return_value = {
                "frame_dict": {1: {0: 0, 1: 1}, 2: {0: 2}},
                "next_tracklet_id": 3,
            }

            result = processor.process_video_observations(mock_video_obs, 10.0, False)

        # Should initialize first frame and process remaining frames
        assert 0 in result  # First frame should be initialized
        assert 1 in result  # Processed frames should be included
        assert 2 in result

        # First frame should map detections to themselves
        assert result[0] == {0: 0, 1: 1}

        # Should call _process_frame_batch once (batch_size=2, processing frames 1-2)
        mock_batch_process.assert_called_once()

    def test_process_video_observations_empty_video(self):
        """Test processing empty video."""
        processor = BatchedFrameProcessor(batch_size=32)

        # Mock video observations with no frames
        mock_video_obs = Mock()
        mock_video_obs._observations = []

        result = processor.process_video_observations(mock_video_obs, 10.0, False)

        # Should return empty dictionary
        assert result == {}

    def test_process_video_observations_single_frame(self):
        """Test processing video with single frame."""
        processor = BatchedFrameProcessor(batch_size=32)

        # Mock video observations with single frame
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock(), Mock(), Mock()]  # Frame 0: 3 detections
        ]

        result = processor.process_video_observations(mock_video_obs, 10.0, False)

        # Should return single frame with identity mapping
        assert result == {0: {0: 0, 1: 1, 2: 2}}

    def test_process_video_observations_two_frames(self):
        """Test processing video with two frames."""
        processor = BatchedFrameProcessor(batch_size=32)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock(), Mock()],  # Frame 0: 2 detections
            [Mock(), Mock()],  # Frame 1: 2 detections
        ]

        # Mock the _process_frame_batch method
        with patch.object(processor, "_process_frame_batch") as mock_batch_process:
            mock_batch_process.return_value = {
                "frame_dict": {1: {0: 0, 1: 1}},
                "next_tracklet_id": 2,
            }

            result = processor.process_video_observations(mock_video_obs, 10.0, False)

        # Should have both frames
        assert len(result) == 2
        assert result[0] == {0: 0, 1: 1}  # First frame identity mapping
        assert result[1] == {0: 0, 1: 1}  # From batch processing

        # Should call batch processing once
        # Note: frame_dict gets updated in-place after the call, so we see the updated version
        mock_batch_process.assert_called_once()
        args = mock_batch_process.call_args[0]
        assert args[0] == mock_video_obs
        assert args[2] == 2  # cur_tracklet_id
        assert args[3] == 1  # batch_start
        assert args[4] == 2  # batch_end
        assert args[5] == 10.0  # max_cost
        assert not args[6]  # rotate_pose

    def test_process_video_observations_batch_processing(self):
        """Test that video is processed in batches."""
        processor = BatchedFrameProcessor(batch_size=2)

        # Mock video observations with 5 frames
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0: 1 detection
            [Mock()],  # Frame 1: 1 detection
            [Mock()],  # Frame 2: 1 detection
            [Mock()],  # Frame 3: 1 detection
            [Mock()],  # Frame 4: 1 detection
        ]

        # Mock the _process_frame_batch method
        with patch.object(processor, "_process_frame_batch") as mock_batch_process:
            mock_batch_process.side_effect = [
                {
                    "frame_dict": {1: {0: 0}, 2: {0: 0}},
                    "next_tracklet_id": 1,
                },  # Batch 1-2
                {
                    "frame_dict": {3: {0: 0}, 4: {0: 0}},
                    "next_tracklet_id": 1,
                },  # Batch 3-4
            ]

            result = processor.process_video_observations(mock_video_obs, 10.0, False)

        # Should process in 2 batches
        assert mock_batch_process.call_count == 2

        # Check batch calls
        calls = mock_batch_process.call_args_list
        assert calls[0][0][3] == 1  # batch_start
        assert calls[0][0][4] == 3  # batch_end
        assert calls[1][0][3] == 3  # batch_start
        assert calls[1][0][4] == 5  # batch_end

        # Should have all frames in result
        assert len(result) == 5
        assert all(frame in result for frame in range(5))

    def test_process_video_observations_parameter_passing(self):
        """Test that parameters are passed correctly to batch processing."""
        processor = BatchedFrameProcessor(batch_size=1)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0
            [Mock()],  # Frame 1
        ]

        # Mock the _process_frame_batch method
        with patch.object(processor, "_process_frame_batch") as mock_batch_process:
            mock_batch_process.return_value = {
                "frame_dict": {1: {0: 0}},
                "next_tracklet_id": 1,
            }

            # Test with custom parameters
            processor.process_video_observations(
                mock_video_obs, max_cost=5.0, rotate_pose=True
            )

        # Check that parameters were passed correctly
        mock_batch_process.assert_called_once()
        args = mock_batch_process.call_args[0]
        assert args[5] == 5.0  # max_cost
        assert args[6]  # rotate_pose

    def test_process_video_observations_tracklet_id_management(self):
        """Test that tracklet IDs are managed correctly across batches."""
        processor = BatchedFrameProcessor(batch_size=1)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock(), Mock()],  # Frame 0: 2 detections
            [Mock()],  # Frame 1: 1 detection
            [Mock(), Mock()],  # Frame 2: 2 detections
        ]

        # Mock the _process_frame_batch method
        with patch.object(processor, "_process_frame_batch") as mock_batch_process:
            mock_batch_process.side_effect = [
                {
                    "frame_dict": {1: {0: 1}},
                    "next_tracklet_id": 3,
                },  # Batch 1, new tracklet created
                {
                    "frame_dict": {2: {0: 1, 1: 3}},
                    "next_tracklet_id": 4,
                },  # Batch 2, another new tracklet
            ]

            processor.process_video_observations(mock_video_obs, 10.0, False)

        # Check that tracklet IDs are passed correctly between batches
        calls = mock_batch_process.call_args_list
        assert calls[0][0][2] == 2  # First batch starts with tracklet ID 2
        assert calls[1][0][2] == 3  # Second batch starts with tracklet ID 3

    def test_process_video_observations_large_batch_size(self):
        """Test processing with large batch size."""
        processor = BatchedFrameProcessor(batch_size=100)

        # Mock video observations with 3 frames
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0
            [Mock()],  # Frame 1
            [Mock()],  # Frame 2
        ]

        # Mock the _process_frame_batch method
        with patch.object(processor, "_process_frame_batch") as mock_batch_process:
            mock_batch_process.return_value = {
                "frame_dict": {1: {0: 0}, 2: {0: 0}},
                "next_tracklet_id": 1,
            }

            processor.process_video_observations(mock_video_obs, 10.0, False)

        # Should process all frames in single batch
        mock_batch_process.assert_called_once()
        args = mock_batch_process.call_args[0]
        assert args[3] == 1  # batch_start
        assert args[4] == 3  # batch_end (all remaining frames)

    def test_process_video_observations_default_parameters(self):
        """Test processing with default parameters."""
        processor = BatchedFrameProcessor()

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0
            [Mock()],  # Frame 1
        ]

        # Mock the _process_frame_batch method
        with patch.object(processor, "_process_frame_batch") as mock_batch_process:
            mock_batch_process.return_value = {
                "frame_dict": {1: {0: 0}},
                "next_tracklet_id": 1,
            }

            processor.process_video_observations(mock_video_obs)

        # Check default parameters
        mock_batch_process.assert_called_once()
        args = mock_batch_process.call_args[0]
        assert args[5] == -np.log(1e-3)  # default max_cost
        assert not args[6]  # default rotate_pose

    def test_process_video_observations_frame_dict_update(self):
        """Test that frame_dict is updated correctly between batches."""
        processor = BatchedFrameProcessor(batch_size=1)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0
            [Mock()],  # Frame 1
            [Mock()],  # Frame 2
        ]

        # Mock the _process_frame_batch method
        with patch.object(processor, "_process_frame_batch") as mock_batch_process:
            mock_batch_process.side_effect = [
                {"frame_dict": {1: {0: 0}}, "next_tracklet_id": 1},
                {"frame_dict": {2: {0: 1}}, "next_tracklet_id": 2},
            ]

            processor.process_video_observations(mock_video_obs, 10.0, False)

        # Check that frame_dict is updated correctly
        calls = mock_batch_process.call_args_list

        # Check that the correct number of calls were made
        assert len(calls) == 2

        # Check the parameters for each call (frame_dict gets updated after each call)
        call1_args = calls[0][0]
        assert call1_args[0] == mock_video_obs
        assert call1_args[2] == 1  # cur_tracklet_id starts at 1
        assert call1_args[3] == 1  # batch_start
        assert call1_args[4] == 2  # batch_end

        call2_args = calls[1][0]
        assert call2_args[0] == mock_video_obs
        assert call2_args[2] == 1  # cur_tracklet_id from first batch result
        assert call2_args[3] == 2  # batch_start
        assert call2_args[4] == 3  # batch_end

    def test_process_video_observations_empty_frames(self):
        """Test processing video with empty frames."""
        processor = BatchedFrameProcessor(batch_size=2)

        # Mock video observations with empty frames
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0: 1 detection
            [],  # Frame 1: 0 detections
            [Mock()],  # Frame 2: 1 detection
        ]

        # Mock the _process_frame_batch method
        with patch.object(processor, "_process_frame_batch") as mock_batch_process:
            mock_batch_process.return_value = {
                "frame_dict": {1: {}, 2: {0: 1}},
                "next_tracklet_id": 2,
            }

            result = processor.process_video_observations(mock_video_obs, 10.0, False)

        # Should handle empty frames correctly
        assert result[0] == {0: 0}  # First frame
        assert result[1] == {}  # Empty frame
        assert result[2] == {0: 1}  # Third frame

    def test_process_video_observations_mixed_frame_sizes(self):
        """Test processing video with varying numbers of detections per frame."""
        processor = BatchedFrameProcessor(batch_size=2)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0: 1 detection
            [Mock(), Mock(), Mock()],  # Frame 1: 3 detections
            [Mock(), Mock()],  # Frame 2: 2 detections
        ]

        # Mock the _process_frame_batch method
        with patch.object(processor, "_process_frame_batch") as mock_batch_process:
            mock_batch_process.return_value = {
                "frame_dict": {1: {0: 0, 1: 1, 2: 2}, 2: {0: 0, 1: 1}},
                "next_tracklet_id": 3,
            }

            result = processor.process_video_observations(mock_video_obs, 10.0, False)

        # Should handle different frame sizes
        assert result[0] == {0: 0}  # 1 detection
        assert result[1] == {0: 0, 1: 1, 2: 2}  # 3 detections
        assert result[2] == {0: 0, 1: 1}  # 2 detections


class TestProcessVideoObservationsEdgeCases:
    """Test edge cases for process_video_observations."""

    def test_process_video_observations_single_detection_per_frame(self):
        """Test processing video with single detection per frame."""
        processor = BatchedFrameProcessor(batch_size=2)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0
            [Mock()],  # Frame 1
            [Mock()],  # Frame 2
        ]

        # Mock the _process_frame_batch method
        with patch.object(processor, "_process_frame_batch") as mock_batch_process:
            mock_batch_process.return_value = {
                "frame_dict": {1: {0: 0}, 2: {0: 0}},
                "next_tracklet_id": 1,
            }

            result = processor.process_video_observations(mock_video_obs, 10.0, False)

        # Should track single detection across frames
        assert all(result[frame] == {0: 0} for frame in range(3))

    def test_process_video_observations_batch_boundary_exact(self):
        """Test processing when frames exactly align with batch boundaries."""
        processor = BatchedFrameProcessor(batch_size=2)

        # Mock video observations (4 frames = 2 batches of 2)
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0
            [Mock()],  # Frame 1
            [Mock()],  # Frame 2
            [Mock()],  # Frame 3
        ]

        # Mock the _process_frame_batch method
        with patch.object(processor, "_process_frame_batch") as mock_batch_process:
            mock_batch_process.side_effect = [
                {"frame_dict": {1: {0: 0}, 2: {0: 0}}, "next_tracklet_id": 1},
                {"frame_dict": {3: {0: 0}}, "next_tracklet_id": 1},
            ]

            processor.process_video_observations(mock_video_obs, 10.0, False)

        # Should process in exactly 2 batches
        assert mock_batch_process.call_count == 2

        # Check batch boundaries
        calls = mock_batch_process.call_args_list
        assert calls[0][0][3:5] == (1, 3)  # First batch: frames 1-2
        assert calls[1][0][3:5] == (3, 4)  # Second batch: frame 3

    def test_process_video_observations_batch_boundary_partial(self):
        """Test processing when last batch is partial."""
        processor = BatchedFrameProcessor(batch_size=3)

        # Mock video observations (4 frames = 1 batch of 3 + 1 partial)
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0
            [Mock()],  # Frame 1
            [Mock()],  # Frame 2
            [Mock()],  # Frame 3
        ]

        # Mock the _process_frame_batch method
        with patch.object(processor, "_process_frame_batch") as mock_batch_process:
            mock_batch_process.side_effect = [
                {
                    "frame_dict": {1: {0: 0}, 2: {0: 0}, 3: {0: 0}},
                    "next_tracklet_id": 1,
                },
            ]

            processor.process_video_observations(mock_video_obs, 10.0, False)

        # Should process in 1 batch (all frames fit)
        assert mock_batch_process.call_count == 1

        # Check batch covers all frames
        calls = mock_batch_process.call_args_list
        assert calls[0][0][3:5] == (1, 4)  # Batch: frames 1-3

    def test_process_video_observations_large_video(self):
        """Test processing large video to verify memory efficiency."""
        processor = BatchedFrameProcessor(batch_size=10)

        # Mock large video observations
        n_frames = 100
        mock_video_obs = Mock()
        mock_video_obs._observations = [[Mock()] for _ in range(n_frames)]

        # Mock the _process_frame_batch method
        with patch.object(processor, "_process_frame_batch") as mock_batch_process:
            mock_batch_process.side_effect = [
                {
                    "frame_dict": {
                        i: {0: 0}
                        for i in range(batch_start, min(batch_start + 10, n_frames))
                    },
                    "next_tracklet_id": 1,
                }
                for batch_start in range(1, n_frames, 10)
            ]

            result = processor.process_video_observations(mock_video_obs, 10.0, False)

        # Should process in multiple batches
        expected_batches = (n_frames - 1 + 9) // 10  # Ceiling division
        assert mock_batch_process.call_count == expected_batches

        # Should have all frames in result
        assert len(result) == n_frames

    def test_process_video_observations_error_propagation(self):
        """Test that errors in batch processing are propagated."""
        processor = BatchedFrameProcessor(batch_size=1)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0
            [Mock()],  # Frame 1
        ]

        # Mock the _process_frame_batch method to raise error
        with patch.object(processor, "_process_frame_batch") as mock_batch_process:
            mock_batch_process.side_effect = RuntimeError("Batch processing error")

            with pytest.raises(RuntimeError, match="Batch processing error"):
                processor.process_video_observations(mock_video_obs, 10.0, False)

    def test_process_video_observations_numerical_parameters(self):
        """Test processing with various numerical parameter values."""
        processor = BatchedFrameProcessor(batch_size=1)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0
            [Mock()],  # Frame 1
        ]

        # Mock the _process_frame_batch method
        with patch.object(processor, "_process_frame_batch") as mock_batch_process:
            mock_batch_process.return_value = {
                "frame_dict": {1: {0: 0}},
                "next_tracklet_id": 1,
            }

            # Test with various max_cost values
            test_costs = [0.1, 1.0, 10.0, 100.0, np.inf]
            for max_cost in test_costs:
                result = processor.process_video_observations(
                    mock_video_obs, max_cost, False
                )
                assert isinstance(result, dict)

            # Test with different rotate_pose values
            for rotate_pose in [True, False]:
                result = processor.process_video_observations(
                    mock_video_obs, 10.0, rotate_pose
                )
                assert isinstance(result, dict)


class TestProcessVideoObservationsIntegration:
    """Test integration scenarios for process_video_observations."""

    def test_process_video_observations_realistic_scenario(self):
        """Test processing with realistic video scenario."""
        processor = BatchedFrameProcessor(batch_size=5)

        # Mock realistic video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock() for _ in range(3)],  # Frame 0: 3 detections
            [Mock() for _ in range(2)],  # Frame 1: 2 detections
            [Mock() for _ in range(4)],  # Frame 2: 4 detections
            [Mock() for _ in range(1)],  # Frame 3: 1 detection
            [Mock() for _ in range(3)],  # Frame 4: 3 detections
        ]

        # Mock the _process_frame_batch method
        with patch.object(processor, "_process_frame_batch") as mock_batch_process:
            mock_batch_process.return_value = {
                "frame_dict": {
                    1: {0: 0, 1: 1},
                    2: {0: 0, 1: 1, 2: 2, 3: 3},
                    3: {0: 0},
                    4: {0: 0, 1: 1, 2: 2},
                },
                "next_tracklet_id": 4,
            }

            result = processor.process_video_observations(mock_video_obs, 5.0, True)

        # Should process all frames
        assert len(result) == 5

        # First frame should be identity mapping
        assert result[0] == {0: 0, 1: 1, 2: 2}

        # Should call batch processing once (all frames fit in one batch)
        mock_batch_process.assert_called_once()

        # Check parameters passed to batch processing
        args = mock_batch_process.call_args[0]
        assert args[5] == 5.0  # max_cost
        assert args[6]  # rotate_pose

    def test_process_video_observations_consistency_across_batch_sizes(self):
        """Test that different batch sizes produce consistent results."""
        # Create processors with different batch sizes
        processor_small = BatchedFrameProcessor(batch_size=1)
        processor_large = BatchedFrameProcessor(batch_size=10)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0
            [Mock()],  # Frame 1
            [Mock()],  # Frame 2
        ]

        # Mock consistent batch processing results
        def mock_batch_process_small(
            video_obs, frame_dict, cur_id, start, end, max_cost, rotate
        ):
            frame_results = {}
            for frame in range(start, end):
                frame_results[frame] = {0: 0}
            return {"frame_dict": frame_results, "next_tracklet_id": cur_id}

        def mock_batch_process_large(
            video_obs, frame_dict, cur_id, start, end, max_cost, rotate
        ):
            frame_results = {}
            for frame in range(start, end):
                frame_results[frame] = {0: 0}
            return {"frame_dict": frame_results, "next_tracklet_id": cur_id}

        # Process with small batch size
        with patch.object(
            processor_small,
            "_process_frame_batch",
            side_effect=mock_batch_process_small,
        ):
            result_small = processor_small.process_video_observations(
                mock_video_obs, 10.0, False
            )

        # Process with large batch size
        with patch.object(
            processor_large,
            "_process_frame_batch",
            side_effect=mock_batch_process_large,
        ):
            result_large = processor_large.process_video_observations(
                mock_video_obs, 10.0, False
            )

        # Results should be consistent
        assert result_small == result_large

    def test_process_video_observations_memory_usage_pattern(self):
        """Test memory usage patterns with different batch sizes."""
        # Test with small batch size (should make more calls)
        processor_small = BatchedFrameProcessor(batch_size=1)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [[Mock()] for _ in range(5)]  # 5 frames

        # Mock the _process_frame_batch method
        with patch.object(
            processor_small, "_process_frame_batch"
        ) as mock_batch_process:
            mock_batch_process.return_value = {
                "frame_dict": {1: {0: 0}},
                "next_tracklet_id": 1,
            }

            processor_small.process_video_observations(mock_video_obs, 10.0, False)

        # Should make 4 calls (frames 1, 2, 3, 4)
        assert mock_batch_process.call_count == 4

        # Test with large batch size (should make fewer calls)
        processor_large = BatchedFrameProcessor(batch_size=10)

        with patch.object(
            processor_large, "_process_frame_batch"
        ) as mock_batch_process:
            mock_batch_process.return_value = {
                "frame_dict": {i: {0: 0} for i in range(1, 5)},
                "next_tracklet_id": 1,
            }

            processor_large.process_video_observations(mock_video_obs, 10.0, False)

        # Should make 1 call (all frames in one batch)
        assert mock_batch_process.call_count == 1
