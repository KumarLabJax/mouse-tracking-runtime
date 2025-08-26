"""Tests for BatchedFrameProcessor class."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from mouse_tracking.matching.batch_processing import BatchedFrameProcessor


class TestBatchedFrameProcessorInit:
    """Test BatchedFrameProcessor initialization."""

    def test_init_default_batch_size(self):
        """Test initialization with default batch size."""
        processor = BatchedFrameProcessor()
        assert processor.batch_size == 32

    def test_init_custom_batch_size(self):
        """Test initialization with custom batch size."""
        processor = BatchedFrameProcessor(batch_size=64)
        assert processor.batch_size == 64

    def test_init_small_batch_size(self):
        """Test initialization with small batch size."""
        processor = BatchedFrameProcessor(batch_size=1)
        assert processor.batch_size == 1

    def test_init_large_batch_size(self):
        """Test initialization with large batch size."""
        processor = BatchedFrameProcessor(batch_size=1000)
        assert processor.batch_size == 1000

    def test_init_batch_size_validation(self):
        """Test that batch size is stored correctly."""
        test_sizes = [1, 2, 8, 16, 32, 64, 128, 256]

        for size in test_sizes:
            processor = BatchedFrameProcessor(batch_size=size)
            assert processor.batch_size == size


class TestBatchedFrameProcessorProcessFrameBatch:
    """Test _process_frame_batch method."""

    def test_process_frame_batch_basic(self):
        """Test basic frame batch processing."""
        processor = BatchedFrameProcessor(batch_size=2)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock(), Mock()],  # Frame 0: 2 detections
            [Mock(), Mock()],  # Frame 1: 2 detections
            [Mock()],  # Frame 2: 1 detection
        ]

        # Mock cost calculation
        mock_video_obs._calculate_costs_vectorized = Mock(
            return_value=np.array([[1.0, 2.0], [3.0, 1.5]])
        )

        # Mock existing frame dict
        frame_dict = {
            0: {0: 0, 1: 1}
        }  # Frame 0 maps detection 0->tracklet 0, detection 1->tracklet 1

        # Mock greedy matching
        with patch(
            "mouse_tracking.matching.batch_processing.vectorized_greedy_matching"
        ) as mock_matching:
            mock_matching.return_value = {0: 0, 1: 1}  # Perfect matching

            result = processor._process_frame_batch(
                mock_video_obs, frame_dict, 2, 1, 3, 10.0, False
            )

        # Check structure
        assert "frame_dict" in result
        assert "next_tracklet_id" in result

        # Check that frames 1 and 2 were processed
        assert 1 in result["frame_dict"]
        assert 2 in result["frame_dict"]

        # Check that tracklet IDs were assigned
        assert result["next_tracklet_id"] >= 2

    def test_process_frame_batch_with_unmatched_detections(self):
        """Test batch processing with unmatched detections."""
        processor = BatchedFrameProcessor(batch_size=1)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock(), Mock()],  # Frame 0: 2 detections
            [Mock(), Mock(), Mock()],  # Frame 1: 3 detections
        ]

        # Mock cost calculation
        mock_video_obs._calculate_costs_vectorized = Mock(
            return_value=np.array([[1.0, 2.0, 5.0], [3.0, 1.5, 4.0]])
        )

        # Mock existing frame dict
        frame_dict = {0: {0: 0, 1: 1}}  # Frame 0 has 2 tracklets

        # Mock greedy matching - only match 2 out of 3 detections
        with patch(
            "mouse_tracking.matching.batch_processing.vectorized_greedy_matching"
        ) as mock_matching:
            mock_matching.return_value = {0: 0, 1: 1}  # Only match first 2

            result = processor._process_frame_batch(
                mock_video_obs, frame_dict, 2, 1, 2, 10.0, False
            )

        # Check that unmatched detection got new tracklet ID
        frame_1_matches = result["frame_dict"][1]
        assert len(frame_1_matches) == 3  # All 3 detections should be assigned
        assert frame_1_matches[0] == 0  # Matched to tracklet 0
        assert frame_1_matches[1] == 1  # Matched to tracklet 1
        assert frame_1_matches[2] == 2  # New tracklet ID for unmatched

        # Check next tracklet ID
        assert result["next_tracklet_id"] == 3

    def test_process_frame_batch_cost_calculation_calls(self):
        """Test that cost calculation is called correctly."""
        processor = BatchedFrameProcessor(batch_size=2)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0: 1 detection
            [Mock()],  # Frame 1: 1 detection
            [Mock()],  # Frame 2: 1 detection
        ]

        # Mock cost calculation
        mock_video_obs._calculate_costs_vectorized = Mock(
            return_value=np.array([[1.0]])
        )

        # Mock existing frame dict
        frame_dict = {0: {0: 0}}

        # Mock greedy matching
        with patch(
            "mouse_tracking.matching.batch_processing.vectorized_greedy_matching"
        ) as mock_matching:
            mock_matching.return_value = {0: 0}

            _ = processor._process_frame_batch(
                mock_video_obs, frame_dict, 1, 1, 3, 10.0, True
            )

        # Check that cost calculation was called for each frame
        assert mock_video_obs._calculate_costs_vectorized.call_count == 2

        # Check the calls were made with correct parameters
        calls = mock_video_obs._calculate_costs_vectorized.call_args_list
        assert calls[0][0] == (0, 1, True)  # (prev_frame, current_frame, rotate_pose)
        assert calls[1][0] == (1, 2, True)

    def test_process_frame_batch_greedy_matching_calls(self):
        """Test that greedy matching is called correctly."""
        processor = BatchedFrameProcessor(batch_size=1)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0: 1 detection
            [Mock()],  # Frame 1: 1 detection
        ]

        # Mock cost calculation
        cost_matrix = np.array([[2.5]])
        mock_video_obs._calculate_costs_vectorized = Mock(return_value=cost_matrix)

        # Mock existing frame dict
        frame_dict = {0: {0: 0}}

        # Mock greedy matching
        with patch(
            "mouse_tracking.matching.batch_processing.vectorized_greedy_matching"
        ) as mock_matching:
            mock_matching.return_value = {0: 0}

            _ = processor._process_frame_batch(
                mock_video_obs, frame_dict, 1, 1, 2, 5.0, False
            )

        # Check that greedy matching was called
        mock_matching.assert_called_once_with(cost_matrix, 5.0)

    def test_process_frame_batch_single_frame(self):
        """Test processing a single frame batch."""
        processor = BatchedFrameProcessor(batch_size=1)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0: 1 detection
            [Mock()],  # Frame 1: 1 detection
        ]

        # Mock cost calculation
        mock_video_obs._calculate_costs_vectorized = Mock(
            return_value=np.array([[1.0]])
        )

        # Mock existing frame dict
        frame_dict = {0: {0: 0}}

        # Mock greedy matching
        with patch(
            "mouse_tracking.matching.batch_processing.vectorized_greedy_matching"
        ) as mock_matching:
            mock_matching.return_value = {0: 0}

            result = processor._process_frame_batch(
                mock_video_obs, frame_dict, 1, 1, 2, 10.0, False
            )

        # Should process only frame 1
        assert len(result["frame_dict"]) == 1
        assert 1 in result["frame_dict"]
        assert result["frame_dict"][1] == {0: 0}

    def test_process_frame_batch_empty_frames(self):
        """Test processing frames with no detections."""
        processor = BatchedFrameProcessor(batch_size=1)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0: 1 detection
            [],  # Frame 1: 0 detections
        ]

        # Mock cost calculation
        mock_video_obs._calculate_costs_vectorized = Mock(
            return_value=np.array([]).reshape(1, 0)
        )

        # Mock existing frame dict
        frame_dict = {0: {0: 0}}

        # Mock greedy matching
        with patch(
            "mouse_tracking.matching.batch_processing.vectorized_greedy_matching"
        ) as mock_matching:
            mock_matching.return_value = {}  # No matches for empty frame

            result = processor._process_frame_batch(
                mock_video_obs, frame_dict, 1, 1, 2, 10.0, False
            )

        # Should process frame 1 with empty matches
        assert len(result["frame_dict"]) == 1
        assert 1 in result["frame_dict"]
        assert result["frame_dict"][1] == {}
        assert result["next_tracklet_id"] == 1  # No new tracklets needed

    def test_process_frame_batch_tracklet_id_continuity(self):
        """Test that tracklet IDs are assigned continuously."""
        processor = BatchedFrameProcessor(batch_size=2)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0: 1 detection
            [Mock(), Mock()],  # Frame 1: 2 detections
            [Mock(), Mock(), Mock()],  # Frame 2: 3 detections
        ]

        # Mock cost calculation
        mock_video_obs._calculate_costs_vectorized = Mock(
            side_effect=[
                np.array([[1.0, 2.0]]),  # Frame 0->1
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # Frame 1->2
            ]
        )

        # Mock existing frame dict
        frame_dict = {0: {0: 0}}  # Start with tracklet 0

        # Mock greedy matching
        with patch(
            "mouse_tracking.matching.batch_processing.vectorized_greedy_matching"
        ) as mock_matching:
            mock_matching.side_effect = [
                {0: 0},  # Frame 1: match detection 0 to prev detection 0
                {0: 0, 1: 1},  # Frame 2: match first 2 detections
            ]

            result = processor._process_frame_batch(
                mock_video_obs, frame_dict, 1, 1, 3, 10.0, False
            )

        # Check frame 1 assignments
        frame_1_matches = result["frame_dict"][1]
        assert frame_1_matches[0] == 0  # Matched to existing tracklet
        assert frame_1_matches[1] == 1  # New tracklet ID

        # Check frame 2 assignments
        frame_2_matches = result["frame_dict"][2]
        assert frame_2_matches[0] == 0  # Matched to existing tracklet
        assert frame_2_matches[1] == 1  # Matched to existing tracklet
        assert frame_2_matches[2] == 2  # New tracklet ID

        # Check next tracklet ID
        assert result["next_tracklet_id"] == 3


class TestBatchedFrameProcessorIntegration:
    """Test integration scenarios for BatchedFrameProcessor."""

    def test_batch_processing_consistency(self):
        """Test that batch processing produces consistent results."""
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

        # Mock cost calculation to return same results
        mock_video_obs._calculate_costs_vectorized = Mock(
            return_value=np.array([[1.0]])
        )

        # Mock existing frame dict
        frame_dict = {0: {0: 0}}

        # Mock greedy matching
        with patch(
            "mouse_tracking.matching.batch_processing.vectorized_greedy_matching"
        ) as mock_matching:
            mock_matching.return_value = {0: 0}

            # Process with small batch size
            result_small = processor_small._process_frame_batch(
                mock_video_obs, frame_dict, 1, 1, 3, 10.0, False
            )

            # Reset mock
            mock_video_obs._calculate_costs_vectorized.reset_mock()
            mock_matching.reset_mock()

            # Process with large batch size
            result_large = processor_large._process_frame_batch(
                mock_video_obs, frame_dict, 1, 1, 3, 10.0, False
            )

        # Results should be the same
        assert result_small["frame_dict"] == result_large["frame_dict"]
        assert result_small["next_tracklet_id"] == result_large["next_tracklet_id"]

    def test_batch_processing_with_different_parameters(self):
        """Test batch processing with different parameter combinations."""
        processor = BatchedFrameProcessor(batch_size=2)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0
            [Mock()],  # Frame 1
        ]

        # Mock cost calculation
        mock_video_obs._calculate_costs_vectorized = Mock(
            return_value=np.array([[1.0]])
        )

        # Mock existing frame dict
        frame_dict = {0: {0: 0}}

        # Test with different rotate_pose values
        with patch(
            "mouse_tracking.matching.batch_processing.vectorized_greedy_matching"
        ) as mock_matching:
            mock_matching.return_value = {0: 0}

            # Test with rotate_pose=False
            _ = processor._process_frame_batch(
                mock_video_obs, frame_dict, 1, 1, 2, 10.0, False
            )

            # Test with rotate_pose=True
            _ = processor._process_frame_batch(
                mock_video_obs, frame_dict, 1, 1, 2, 10.0, True
            )

        # Check that cost calculation was called with correct rotate_pose parameter
        calls = mock_video_obs._calculate_costs_vectorized.call_args_list
        assert calls[0][0][2] is False  # First call with rotate_pose=False
        assert calls[1][0][2] is True  # Second call with rotate_pose=True

    def test_batch_processing_memory_efficiency(self):
        """Test that batch processing doesn't accumulate unnecessary data."""
        processor = BatchedFrameProcessor(batch_size=1)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0
            [Mock()],  # Frame 1
        ]

        # Mock cost calculation
        mock_video_obs._calculate_costs_vectorized = Mock(
            return_value=np.array([[1.0]])
        )

        # Mock existing frame dict
        frame_dict = {0: {0: 0}}

        # Mock greedy matching
        with patch(
            "mouse_tracking.matching.batch_processing.vectorized_greedy_matching"
        ) as mock_matching:
            mock_matching.return_value = {0: 0}

            result = processor._process_frame_batch(
                mock_video_obs, frame_dict, 1, 1, 2, 10.0, False
            )

        # Result should only contain the processed frames
        assert len(result["frame_dict"]) == 1
        assert 1 in result["frame_dict"]
        assert 0 not in result["frame_dict"]  # Previous frame not included

    def test_batch_size_boundary_conditions(self):
        """Test batch processing at boundary conditions."""
        # Test with batch size equal to number of frames
        processor = BatchedFrameProcessor(batch_size=2)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0
            [Mock()],  # Frame 1
            [Mock()],  # Frame 2
        ]

        # Mock cost calculation
        mock_video_obs._calculate_costs_vectorized = Mock(
            return_value=np.array([[1.0]])
        )

        # Mock existing frame dict
        frame_dict = {0: {0: 0}}

        # Mock greedy matching
        with patch(
            "mouse_tracking.matching.batch_processing.vectorized_greedy_matching"
        ) as mock_matching:
            mock_matching.return_value = {0: 0}

            # Process exactly 2 frames (batch_size)
            result = processor._process_frame_batch(
                mock_video_obs, frame_dict, 1, 1, 3, 10.0, False
            )

        # Should process both frames
        assert len(result["frame_dict"]) == 2
        assert 1 in result["frame_dict"]
        assert 2 in result["frame_dict"]

    def test_error_handling_in_batch_processing(self):
        """Test error handling during batch processing."""
        processor = BatchedFrameProcessor(batch_size=1)

        # Mock video observations
        mock_video_obs = Mock()
        mock_video_obs._observations = [
            [Mock()],  # Frame 0
            [Mock()],  # Frame 1
        ]

        # Mock cost calculation to raise an error
        mock_video_obs._calculate_costs_vectorized = Mock(
            side_effect=RuntimeError("Test error")
        )

        # Mock existing frame dict
        frame_dict = {0: {0: 0}}

        # Should propagate the error
        with pytest.raises(RuntimeError, match="Test error"):
            processor._process_frame_batch(
                mock_video_obs, frame_dict, 1, 1, 2, 10.0, False
            )
