"""Comprehensive unit tests for the v2_to_v3 pose conversion function."""

import numpy as np
import pytest

from mouse_tracking.pose.convert import v2_to_v3


class TestV2ToV3BasicFunctionality:
    """Test basic functionality and successful conversions."""

    def test_basic_conversion_all_good_data(self):
        """Test basic conversion with all confidence values above threshold."""
        # Arrange
        pose_data = (
            np.random.rand(10, 12, 2) * 100
        )  # 10 frames, 12 keypoints, x,y coords
        conf_data = np.full((10, 12), 0.8)  # All confidence above default threshold 0.3
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        # Check shapes
        assert pose_data_v3.shape == (10, 1, 12, 2)
        assert conf_data_v3.shape == (10, 1, 12)
        assert instance_count.shape == (10,)
        assert instance_embedding.shape == (10, 1, 12)
        assert instance_track_id.shape == (10, 1)

        # Check data types
        assert pose_data_v3.dtype == pose_data.dtype
        assert conf_data_v3.dtype == conf_data.dtype
        assert instance_count.dtype == np.uint8
        assert instance_embedding.dtype == np.float32
        assert instance_track_id.dtype == np.uint32

        # Check values
        np.testing.assert_array_equal(pose_data_v3[:, 0, :, :], pose_data)
        np.testing.assert_array_equal(conf_data_v3[:, 0, :], conf_data)
        np.testing.assert_array_equal(instance_count, np.ones(10, dtype=np.uint8))
        np.testing.assert_array_equal(
            instance_embedding, np.zeros((10, 1, 12), dtype=np.float32)
        )
        np.testing.assert_array_equal(
            instance_track_id, np.zeros((10, 1), dtype=np.uint32)
        )

    def test_basic_conversion_with_bad_data(self):
        """Test conversion with some confidence values below threshold."""
        # Arrange
        pose_data = np.ones((5, 12, 2)) * 10
        conf_data = np.array(
            [
                [
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.2,
                    0.2,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                ],  # Some low confidence
                [
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                ],  # All good
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # All bad
                [
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                ],  # All good
                [
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                ],  # Some good
            ]
        )
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        # Frame 0: has some good keypoints, should have instance_count = 1
        # Frame 1: all good keypoints, should have instance_count = 1
        # Frame 2: all bad keypoints, should have instance_count = 0
        # Frame 3: all good keypoints, should have instance_count = 1
        # Frame 4: some good keypoints, should have instance_count = 1
        expected_instance_count = np.array([1, 1, 0, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(instance_count, expected_instance_count)

        # Check that bad pose data is zeroed out
        bad_pose_mask = conf_data_v3 < threshold
        assert np.all(pose_data_v3[bad_pose_mask] == 0)
        assert np.all(conf_data_v3[bad_pose_mask] == 0)

        # Check track IDs - should be 0 for first segment, then 1 for segment after gap
        expected_track_ids = np.array([[0], [0], [0], [1], [1]], dtype=np.uint32)
        np.testing.assert_array_equal(instance_track_id, expected_track_ids)

    def test_conversion_preserves_good_pose_data(self):
        """Test that pose data above threshold is preserved unchanged."""
        # Arrange
        pose_data = np.array(
            [
                [
                    [1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8],
                    [9, 10],
                    [11, 12],
                    [13, 14],
                    [15, 16],
                    [17, 18],
                    [19, 20],
                    [21, 22],
                    [23, 24],
                ],
                [
                    [25, 26],
                    [27, 28],
                    [29, 30],
                    [31, 32],
                    [33, 34],
                    [35, 36],
                    [37, 38],
                    [39, 40],
                    [41, 42],
                    [43, 44],
                    [45, 46],
                    [47, 48],
                ],
            ]
        )
        conf_data = np.full((2, 12), 0.8)  # All above threshold
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        # Good data should be preserved
        np.testing.assert_array_equal(pose_data_v3[:, 0, :, :], pose_data)
        np.testing.assert_array_equal(conf_data_v3[:, 0, :], conf_data)

    @pytest.mark.parametrize(
        "threshold,expected_instance_counts",
        [
            (0.1, [1, 1, 1, 1]),  # Very low threshold - all frames valid
            (0.4, [1, 1, 0, 1]),  # Medium threshold - frame 2 invalid
            (0.6, [1, 1, 0, 0]),  # High threshold - frames 2,3 invalid
            (0.9, [0, 0, 0, 0]),  # Very high threshold - all frames invalid
        ],
        ids=[
            "very_low_threshold",
            "medium_threshold",
            "high_threshold",
            "very_high_threshold",
        ],
    )
    def test_different_thresholds(self, threshold, expected_instance_counts):
        """Test conversion with different confidence thresholds."""
        # Arrange
        pose_data = np.ones((4, 12, 2)) * 10
        conf_data = np.array(
            [
                [0.8] * 12,  # High confidence
                [0.7] * 12,  # Medium-high confidence
                [0.2] * 12,  # Low confidence
                [0.5] * 12,  # Medium confidence
            ]
        )

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        np.testing.assert_array_equal(
            instance_count, np.array(expected_instance_counts, dtype=np.uint8)
        )


class TestV2ToV3TrackletGeneration:
    """Test tracklet ID generation from run-length encoding."""

    def test_continuous_valid_frames_single_tracklet(self):
        """Test that continuous valid frames get a single tracklet ID."""
        # Arrange
        pose_data = np.ones((5, 12, 2))
        conf_data = np.full((5, 12), 0.8)
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        expected_track_ids = np.zeros((5, 1), dtype=np.uint32)
        np.testing.assert_array_equal(instance_track_id, expected_track_ids)

    def test_discontinuous_segments_multiple_tracklets(self):
        """Test that discontinuous segments get different tracklet IDs."""
        # Arrange
        pose_data = np.ones((7, 12, 2))
        conf_data = np.array(
            [
                [0.8] * 12,  # Frame 0: valid -> tracklet 0
                [0.8] * 12,  # Frame 1: valid -> tracklet 0
                [0.1] * 12,  # Frame 2: invalid -> no tracklet
                [0.1] * 12,  # Frame 3: invalid -> no tracklet
                [0.8] * 12,  # Frame 4: valid -> tracklet 1
                [0.8] * 12,  # Frame 5: valid -> tracklet 1
                [0.8] * 12,  # Frame 6: valid -> tracklet 1
            ]
        )
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        expected_track_ids = np.array(
            [[0], [0], [0], [0], [1], [1], [1]], dtype=np.uint32
        )
        np.testing.assert_array_equal(instance_track_id, expected_track_ids)

    def test_multiple_short_segments(self):
        """Test multiple short valid segments get incrementing tracklet IDs."""
        # Arrange
        pose_data = np.ones((9, 12, 2))
        conf_data = np.array(
            [
                [0.8] * 12,  # Frame 0: valid -> tracklet 0
                [0.1] * 12,  # Frame 1: invalid -> tracklet 0 (gap)
                [0.8] * 12,  # Frame 2: valid -> tracklet 1
                [0.1] * 12,  # Frame 3: invalid -> tracklet 0 (gap)
                [0.8] * 12,  # Frame 4: valid -> tracklet 2
                [0.8] * 12,  # Frame 5: valid -> tracklet 2
                [0.1] * 12,  # Frame 6: invalid -> tracklet 0 (gap)
                [0.8] * 12,  # Frame 7: valid -> tracklet 3
                [0.1] * 12,  # Frame 8: invalid -> tracklet 0 (gap)
            ]
        )
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        # Expected instance_count: [1, 0, 1, 0, 1, 1, 0, 1, 0]
        # Expected track_ids: [0, 0, 1, 0, 2, 2, 0, 3, 0] (invalid frames get tracklet 0)
        expected_instance_count = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
        expected_track_ids = np.array(
            [[0], [0], [1], [0], [2], [2], [0], [3], [0]], dtype=np.uint32
        )
        np.testing.assert_array_equal(instance_count, expected_instance_count)
        np.testing.assert_array_equal(instance_track_id, expected_track_ids)


class TestV2ToV3EdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_arrays(self):
        """Test conversion with empty input arrays."""
        # Arrange
        pose_data = np.empty((0, 12, 2))
        conf_data = np.empty((0, 12))
        threshold = 0.3

        # Act & Assert
        # NOTE: This currently fails due to a bug in the implementation
        # where run_length_encode returns None for empty arrays, but the code
        # tries to subscript it. This should be fixed in the implementation.
        with pytest.raises(TypeError, match="'NoneType' object is not subscriptable"):
            (
                pose_data_v3,
                conf_data_v3,
                instance_count,
                instance_embedding,
                instance_track_id,
            ) = v2_to_v3(pose_data, conf_data, threshold)

    def test_single_frame_valid(self):
        """Test conversion with a single valid frame."""
        # Arrange
        pose_data = np.ones((1, 12, 2)) * 5
        conf_data = np.full((1, 12), 0.8)
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        assert pose_data_v3.shape == (1, 1, 12, 2)
        np.testing.assert_array_equal(instance_count, np.array([1], dtype=np.uint8))
        np.testing.assert_array_equal(
            instance_track_id, np.array([[0]], dtype=np.uint32)
        )
        np.testing.assert_array_equal(pose_data_v3[0, 0, :, :], pose_data[0, :, :])

    def test_single_frame_invalid(self):
        """Test conversion with a single invalid frame."""
        # Arrange
        pose_data = np.ones((1, 12, 2)) * 5
        conf_data = np.full((1, 12), 0.1)  # Below threshold
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        assert pose_data_v3.shape == (1, 1, 12, 2)
        np.testing.assert_array_equal(instance_count, np.array([0], dtype=np.uint8))
        np.testing.assert_array_equal(
            instance_track_id, np.array([[0]], dtype=np.uint32)
        )
        # Pose data should be zeroed out
        np.testing.assert_array_equal(pose_data_v3[0, 0, :, :], np.zeros((12, 2)))

    def test_all_frames_invalid(self):
        """Test conversion where all frames have confidence below threshold."""
        # Arrange
        pose_data = np.ones((5, 12, 2)) * 10
        conf_data = np.full((5, 12), 0.1)  # All below threshold
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        np.testing.assert_array_equal(instance_count, np.zeros(5, dtype=np.uint8))
        np.testing.assert_array_equal(pose_data_v3, np.zeros((5, 1, 12, 2)))
        np.testing.assert_array_equal(conf_data_v3, np.zeros((5, 1, 12)))
        # All frames invalid, so all track IDs should be 0
        np.testing.assert_array_equal(
            instance_track_id, np.zeros((5, 1), dtype=np.uint32)
        )

    def test_partial_keypoint_filtering(self):
        """Test that only specific keypoints below threshold are filtered."""
        # Arrange
        pose_data = np.ones((2, 12, 2)) * 10
        conf_data = np.array(
            [
                [
                    0.8,
                    0.8,
                    0.1,
                    0.1,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                ],  # Keypoints 2,3 low
                [
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                    0.1,
                    0.1,
                    0.8,
                    0.8,
                    0.8,
                    0.8,
                ],  # Keypoints 6,7 low
            ]
        )
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        # Both frames should be valid (have some good keypoints)
        np.testing.assert_array_equal(instance_count, np.array([1, 1], dtype=np.uint8))

        # Check that only specific keypoints are zeroed
        assert np.all(
            pose_data_v3[0, 0, [2, 3], :] == 0
        )  # Frame 0, keypoints 2,3 zeroed
        assert np.all(
            pose_data_v3[0, 0, [0, 1, 4, 5, 6, 7, 8, 9, 10, 11], :] == 10
        )  # Other keypoints preserved

        assert np.all(
            pose_data_v3[1, 0, [6, 7], :] == 0
        )  # Frame 1, keypoints 6,7 zeroed
        assert np.all(
            pose_data_v3[1, 0, [0, 1, 2, 3, 4, 5, 8, 9, 10, 11], :] == 10
        )  # Other keypoints preserved

    @pytest.mark.parametrize(
        "threshold",
        [0.0, 1.0, 0.5, 0.001, 0.999],
        ids=[
            "zero_threshold",
            "max_threshold",
            "half_threshold",
            "very_low_threshold",
            "very_high_threshold",
        ],
    )
    def test_boundary_thresholds(self, threshold):
        """Test conversion with boundary threshold values."""
        # Arrange
        pose_data = np.ones((3, 12, 2))
        conf_data = np.array(
            [
                [0.0] * 12,  # Exactly zero confidence
                [0.5] * 12,  # Middle confidence
                [1.0] * 12,  # Maximum confidence
            ]
        )

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        # Should not raise any errors and produce valid output shapes
        assert pose_data_v3.shape == (3, 1, 12, 2)
        assert conf_data_v3.shape == (3, 1, 12)
        assert instance_count.shape == (3,)
        assert instance_embedding.shape == (3, 1, 12)
        assert instance_track_id.shape == (3, 1)

        # Verify filtering logic
        for frame_idx in range(3):
            frame_conf = conf_data[frame_idx]
            valid_keypoints = np.sum(frame_conf >= threshold)
            if valid_keypoints > 0:
                assert instance_count[frame_idx] == 1
            else:
                assert instance_count[frame_idx] == 0


class TestV2ToV3DataTypes:
    """Test data type handling and preservation."""

    @pytest.mark.parametrize(
        "pose_dtype,conf_dtype",
        [
            (np.float32, np.float32),
            (np.float64, np.float64),
            (np.float32, np.float64),
            (np.float64, np.float32),
        ],
        ids=[
            "both_float32",
            "both_float64",
            "pose_float32_conf_float64",
            "pose_float64_conf_float32",
        ],
    )
    def test_data_type_preservation(self, pose_dtype, conf_dtype):
        """Test that input data types are preserved in output."""
        # Arrange
        pose_data = np.ones((3, 12, 2), dtype=pose_dtype)
        conf_data = np.full((3, 12), 0.8, dtype=conf_dtype)
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        assert pose_data_v3.dtype == pose_dtype
        assert conf_data_v3.dtype == conf_dtype
        assert instance_count.dtype == np.uint8
        assert instance_embedding.dtype == np.float32
        assert instance_track_id.dtype == np.uint32

    def test_integer_pose_data(self):
        """Test conversion with integer pose data."""
        # Arrange
        pose_data = np.ones((2, 12, 2), dtype=np.int32) * 10
        conf_data = np.full((2, 12), 0.8, dtype=np.float32)
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        assert pose_data_v3.dtype == np.int32
        assert conf_data_v3.dtype == np.float32


class TestV2ToV3ErrorHandling:
    """Test error conditions and invalid inputs."""

    def test_mismatched_array_shapes(self):
        """Test error handling with mismatched input array shapes."""
        # Arrange
        pose_data = np.ones((5, 12, 2))
        conf_data = np.ones((3, 12))  # Different number of frames
        threshold = 0.3

        # Act & Assert
        # The function doesn't validate input shapes properly and fails during boolean indexing
        with pytest.raises(
            IndexError, match="boolean index did not match indexed array"
        ):
            v2_to_v3(pose_data, conf_data, threshold)

    def test_wrong_pose_data_dimensions(self):
        """Test error handling with incorrect pose data dimensions."""
        # Arrange
        pose_data = np.ones((5, 12))  # Missing coordinate dimension
        conf_data = np.ones((5, 12))
        threshold = 0.3

        # Act & Assert
        with pytest.raises((ValueError, IndexError)):
            v2_to_v3(pose_data, conf_data, threshold)

    def test_wrong_confidence_dimensions(self):
        """Test error handling with incorrect confidence data dimensions."""
        # Arrange
        pose_data = np.ones((5, 12, 2))
        conf_data = np.ones((5, 12, 2))  # Extra dimension
        threshold = 0.3

        # Act & Assert
        with pytest.raises((ValueError, IndexError)):
            v2_to_v3(pose_data, conf_data, threshold)

    def test_negative_threshold(self):
        """Test conversion with negative threshold."""
        # Arrange
        pose_data = np.ones((2, 12, 2))
        conf_data = np.full((2, 12), 0.5)
        threshold = -0.1

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        # Should work (all confidence values > negative threshold)
        np.testing.assert_array_equal(instance_count, np.array([1, 1], dtype=np.uint8))

    def test_very_large_threshold(self):
        """Test conversion with threshold larger than 1.0."""
        # Arrange
        pose_data = np.ones((2, 12, 2))
        conf_data = np.full((2, 12), 0.9)
        threshold = 2.0

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        # All confidence values should be below threshold
        np.testing.assert_array_equal(instance_count, np.array([0, 0], dtype=np.uint8))


class TestV2ToV3LargeDatasets:
    """Test performance and correctness with larger datasets."""

    def test_large_dataset_conversion(self):
        """Test conversion with a large dataset to ensure scalability."""
        # Arrange
        num_frames = 1000
        pose_data = np.random.rand(num_frames, 12, 2) * 100
        conf_data = np.random.rand(num_frames, 12)
        threshold = 0.5

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        assert pose_data_v3.shape == (num_frames, 1, 12, 2)
        assert conf_data_v3.shape == (num_frames, 1, 12)
        assert instance_count.shape == (num_frames,)
        assert instance_embedding.shape == (num_frames, 1, 12)
        assert instance_track_id.shape == (num_frames, 1)

        # Verify that filtering was applied correctly
        bad_pose_mask = conf_data_v3 < threshold
        assert np.all(pose_data_v3[bad_pose_mask] == 0)
        assert np.all(conf_data_v3[bad_pose_mask] == 0)

    def test_memory_efficiency_large_arrays(self):
        """Test that function doesn't create unnecessary large intermediate arrays."""
        # Arrange
        num_frames = 10000  # Large dataset
        pose_data = np.ones((num_frames, 12, 2), dtype=np.float32)
        conf_data = np.full((num_frames, 12), 0.8, dtype=np.float32)
        threshold = 0.3

        # Act (should complete without memory errors)
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        assert pose_data_v3.shape == (num_frames, 1, 12, 2)
        # Verify all instances are valid (all confidence above threshold)
        assert np.all(instance_count == 1)


class TestV2ToV3SpecialValues:
    """Test handling of special floating point values."""

    def test_nan_confidence_values(self):
        """Test handling of NaN confidence values."""
        # Arrange
        pose_data = np.ones((3, 12, 2))
        conf_data = np.array(
            [
                [0.8, 0.8, np.nan, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                [np.nan] * 12,
            ]
        )
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        # NOTE: NaN < threshold returns False, so NaN keypoints are NOT filtered out
        # This means frames with NaN confidence are still considered valid instances
        # Frame 0: has valid keypoints (including NaN), should be valid
        # Frame 1: all valid keypoints, should be valid
        # Frame 2: all NaN (which are not < threshold), should be valid
        #
        # TODO: (From Brian) - "Not sure I agree with this behavior, but I don't think
        #   it affects any data. NAN confidence should probably be filtered out."
        expected_instance_count = np.array([1, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(instance_count, expected_instance_count)

    def test_infinity_confidence_values(self):
        """Test handling of infinity confidence values."""
        # Arrange
        pose_data = np.ones((2, 12, 2))
        conf_data = np.array(
            [
                [0.8, np.inf, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                [-np.inf, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            ]
        )
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        # inf > threshold, so those keypoints should be preserved
        # -inf < threshold, so those keypoints should be filtered
        expected_instance_count = np.array([1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(instance_count, expected_instance_count)

        # Check specific filtering
        assert conf_data_v3[1, 0, 0] == 0  # -inf should be filtered to 0
        assert conf_data_v3[0, 0, 1] == np.inf  # +inf should be preserved

    def test_confidence_values_greater_than_one(self):
        """Test handling of confidence values greater than 1.0 (realistic HRNet output)."""
        # Arrange
        pose_data = np.ones((4, 12, 2)) * 50
        conf_data = np.array(
            [
                [1.1] * 12,  # Slightly above 1.0
                [1.5] * 12,  # Moderately above 1.0
                [2.3] * 12,  # Well above 1.0
                [0.5, 1.2, 0.8, 2.1, 0.3, 1.0, 0.9, 1.8, 0.2, 1.5, 0.7, 2.0],  # Mixed
            ]
        )
        threshold = 0.6

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        # All frames should be valid since values > 1.0 are > threshold
        expected_instance_count = np.array([1, 1, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(instance_count, expected_instance_count)

        # Check that values > 1.0 are preserved as-is
        np.testing.assert_array_equal(conf_data_v3[0, 0, :], [1.1] * 12)
        np.testing.assert_array_equal(conf_data_v3[1, 0, :], [1.5] * 12)
        np.testing.assert_array_equal(conf_data_v3[2, 0, :], [2.3] * 12)

        # Check mixed frame filtering (only values < threshold should be zeroed)
        expected_mixed_frame = np.array(
            [0.0, 1.2, 0.8, 2.1, 0.0, 1.0, 0.9, 1.8, 0.0, 1.5, 0.7, 2.0]
        )
        np.testing.assert_array_equal(conf_data_v3[3, 0, :], expected_mixed_frame)

    def test_negative_confidence_values(self):
        """Test handling of negative confidence values (possible HRNet output)."""
        # Arrange
        pose_data = np.ones((4, 12, 2)) * 25
        conf_data = np.array(
            [
                [-0.1] * 12,  # Slightly negative
                [-0.5] * 12,  # Moderately negative
                [-2.0] * 12,  # Very negative
                [
                    0.8,
                    -0.2,
                    0.9,
                    -0.1,
                    0.7,
                    -0.3,
                    0.6,
                    -0.4,
                    0.5,
                    -0.5,
                    0.4,
                    -0.6,
                ],  # Mixed
            ]
        )
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        # First three frames should be invalid (all negative < threshold)
        # Fourth frame should be valid (has some values >= threshold)
        expected_instance_count = np.array([0, 0, 0, 1], dtype=np.uint8)
        np.testing.assert_array_equal(instance_count, expected_instance_count)

        # Check that negative values are filtered to 0
        np.testing.assert_array_equal(conf_data_v3[0, 0, :], np.zeros(12))
        np.testing.assert_array_equal(conf_data_v3[1, 0, :], np.zeros(12))
        np.testing.assert_array_equal(conf_data_v3[2, 0, :], np.zeros(12))

        # Check mixed frame filtering
        expected_mixed_frame = np.array(
            [0.8, 0.0, 0.9, 0.0, 0.7, 0.0, 0.6, 0.0, 0.5, 0.0, 0.4, 0.0]
        )
        np.testing.assert_array_equal(conf_data_v3[3, 0, :], expected_mixed_frame)

        # Corresponding pose data should also be zeroed for filtered keypoints
        for frame_idx in range(3):
            np.testing.assert_array_equal(
                pose_data_v3[frame_idx, 0, :, :], np.zeros((12, 2))
            )

    def test_extreme_out_of_bounds_confidence_values(self):
        """Test handling of extremely out-of-bounds confidence values."""
        # Arrange
        pose_data = np.ones((3, 12, 2)) * 100
        conf_data = np.array(
            [
                [
                    10.0,
                    -5.0,
                    0.5,
                    100.0,
                    -10.0,
                    0.8,
                    50.0,
                    -1.0,
                    0.3,
                    200.0,
                    -20.0,
                    0.1,
                ],
                [1000.0] * 12,  # Very large positive values
                [-1000.0] * 12,  # Very large negative values
            ]
        )
        threshold = 0.4

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        expected_instance_count = np.array([1, 1, 0], dtype=np.uint8)
        np.testing.assert_array_equal(instance_count, expected_instance_count)

        # Check extreme positive values are preserved
        np.testing.assert_array_equal(conf_data_v3[1, 0, :], [1000.0] * 12)

        # Check extreme negative values are filtered
        np.testing.assert_array_equal(conf_data_v3[2, 0, :], np.zeros(12))

        # Check mixed extreme values
        expected_mixed = np.array(
            [10.0, 0.0, 0.5, 100.0, 0.0, 0.8, 50.0, 0.0, 0.0, 200.0, 0.0, 0.0]
        )
        np.testing.assert_array_equal(conf_data_v3[0, 0, :], expected_mixed)


class TestV2ToV3ComprehensiveScenarios:
    """Test comprehensive real-world scenarios that might occur during refactoring."""

    def test_alternating_valid_invalid_pattern(self):
        """Test alternating valid/invalid frames pattern."""
        # Arrange
        pose_data = np.ones((6, 12, 2)) * 50
        conf_data = np.array(
            [
                [0.8] * 12,  # Frame 0: valid -> tracklet 0
                [0.1] * 12,  # Frame 1: invalid
                [0.8] * 12,  # Frame 2: valid -> tracklet 1
                [0.1] * 12,  # Frame 3: invalid
                [0.8] * 12,  # Frame 4: valid -> tracklet 2
                [0.1] * 12,  # Frame 5: invalid
            ]
        )
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        expected_instance_count = np.array([1, 0, 1, 0, 1, 0], dtype=np.uint8)
        expected_track_ids = np.array([[0], [0], [1], [0], [2], [0]], dtype=np.uint32)
        np.testing.assert_array_equal(instance_count, expected_instance_count)
        np.testing.assert_array_equal(instance_track_id, expected_track_ids)

    def test_confidence_exactly_at_threshold(self):
        """Test behavior when confidence values are exactly at threshold."""
        # Arrange
        pose_data = np.ones((3, 12, 2)) * 10
        threshold = 0.5
        conf_data = np.array(
            [
                [0.5] * 12,  # Exactly at threshold
                [0.49999] * 12,  # Just below threshold
                [0.50001] * 12,  # Just above threshold
            ]
        )

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        # conf >= threshold should be preserved, conf < threshold should be filtered
        expected_instance_count = np.array([1, 0, 1], dtype=np.uint8)
        np.testing.assert_array_equal(instance_count, expected_instance_count)

        # Check filtering
        assert np.all(conf_data_v3[0, 0, :] == 0.5)  # Exactly at threshold preserved
        assert np.all(conf_data_v3[1, 0, :] == 0)  # Below threshold filtered
        assert np.all(conf_data_v3[2, 0, :] == 0.50001)  # Above threshold preserved

    def test_mixed_keypoint_confidence_realistic(self):
        """Test realistic scenario with mixed keypoint confidence."""
        # Arrange
        pose_data = np.random.rand(5, 12, 2) * 200
        # Simulate realistic confidence patterns
        conf_data = np.array(
            [
                # Frame 0: nose and ears high conf, body parts medium, tail low
                [0.9, 0.8, 0.85, 0.6, 0.4, 0.45, 0.7, 0.3, 0.25, 0.2, 0.15, 0.1],
                # Frame 1: mostly good confidence
                [0.8, 0.75, 0.8, 0.7, 0.6, 0.65, 0.8, 0.5, 0.45, 0.4, 0.35, 0.3],
                # Frame 2: poor tracking quality
                [0.2, 0.15, 0.1, 0.05, 0.1, 0.15, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0],
                # Frame 3: back to good quality
                [0.85, 0.8, 0.9, 0.75, 0.7, 0.65, 0.8, 0.6, 0.55, 0.5, 0.45, 0.4],
                # Frame 4: partial occlusion (some keypoints invisible)
                [0.9, 0.85, 0.8, 0.1, 0.05, 0.1, 0.75, 0.7, 0.65, 0.0, 0.0, 0.0],
            ]
        )
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        # Check that frames with at least some good keypoints are valid
        # Check that low confidence keypoints are filtered individually
        for frame in range(5):
            valid_keypoints = np.sum(conf_data[frame] >= threshold)
            if valid_keypoints > 0:
                assert instance_count[frame] == 1
            else:
                assert instance_count[frame] == 0

            # Check that low confidence keypoints are zeroed
            low_conf_mask = conf_data[frame] < threshold
            assert np.all(conf_data_v3[frame, 0, low_conf_mask] == 0)
            assert np.all(pose_data_v3[frame, 0, low_conf_mask, :] == 0)

    def test_long_sequence_with_gaps(self):
        """Test long sequence with various gap patterns."""
        # Arrange
        num_frames = 50
        pose_data = np.ones((num_frames, 12, 2))
        conf_data = np.full((num_frames, 12), 0.1)  # Start with all low confidence

        # Add valid segments at specific intervals
        valid_segments = [
            (0, 5),  # tracklet 0: frames 0-4
            (10, 15),  # tracklet 1: frames 10-14
            (20, 25),  # tracklet 2: frames 20-24
            (30, 40),  # tracklet 3: frames 30-39
            (45, 50),  # tracklet 4: frames 45-49
        ]

        for start, end in valid_segments:
            conf_data[start:end] = 0.8

        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        # Check that each valid segment gets a unique tracklet ID
        for tracklet_counter, (start, end) in enumerate(valid_segments):
            # All frames in this segment should have the same tracklet ID
            segment_track_ids = instance_track_id[start:end, 0]
            assert np.all(segment_track_ids == tracklet_counter)

            # All frames in this segment should be valid
            assert np.all(instance_count[start:end] == 1)

        # Check that gap frames are invalid
        for i in range(num_frames):
            in_valid_segment = any(start <= i < end for start, end in valid_segments)
            if not in_valid_segment:
                assert instance_count[i] == 0

    def test_zero_confidence_boundary_case(self):
        """Test edge case with exactly zero confidence values."""
        # Arrange
        pose_data = np.ones((3, 12, 2)) * 100
        conf_data = np.array(
            [
                [0.0] * 12,  # All exactly zero
                [0.0] * 6 + [0.5] * 6,  # Half zero, half above threshold
                [0.5] * 6 + [0.0] * 6,  # Half above threshold, half zero
            ]
        )
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        expected_instance_count = np.array([0, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(instance_count, expected_instance_count)

        # Check zero filtering
        assert np.all(conf_data_v3[0, 0, :] == 0)  # All zeros stay zero
        assert np.all(pose_data_v3[0, 0, :, :] == 0)  # Corresponding poses zeroed

    def test_non_standard_keypoint_count_error(self):
        """Test that function only works with 12 keypoints (implementation constraint)."""
        # Arrange
        pose_data_wrong_size = np.ones((3, 6, 2)) * 10  # 6 keypoints instead of 12
        conf_data_wrong_size = np.full((3, 6), 0.8)
        threshold = 0.3

        # Act & Assert
        # The function is hardcoded for 12 keypoints and will fail with other sizes
        with pytest.raises(ValueError, match="cannot reshape array"):
            v2_to_v3(pose_data_wrong_size, conf_data_wrong_size, threshold)

    def test_standard_12_keypoints_works(self):
        """Test that function works correctly with standard 12 keypoints."""
        # Arrange
        pose_data = np.ones((3, 12, 2)) * 10
        conf_data = np.full((3, 12), 0.8)
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        assert pose_data_v3.shape == (3, 1, 12, 2)
        assert conf_data_v3.shape == (3, 1, 12)
        assert instance_embedding.shape == (3, 1, 12)
        np.testing.assert_array_equal(instance_count, np.ones(3, dtype=np.uint8))

    def test_very_small_pose_coordinates(self):
        """Test with very small pose coordinate values."""
        # Arrange
        pose_data = np.ones((2, 12, 2)) * 1e-10  # Very small coordinates
        conf_data = np.full((2, 12), 0.8)
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        np.testing.assert_array_almost_equal(pose_data_v3[:, 0, :, :], pose_data)
        np.testing.assert_array_equal(instance_count, np.ones(2, dtype=np.uint8))

    def test_very_large_pose_coordinates(self):
        """Test with very large pose coordinate values."""
        # Arrange
        pose_data = np.ones((2, 12, 2)) * 1e6  # Very large coordinates
        conf_data = np.full((2, 12), 0.8)
        threshold = 0.3

        # Act
        (
            pose_data_v3,
            conf_data_v3,
            instance_count,
            instance_embedding,
            instance_track_id,
        ) = v2_to_v3(pose_data, conf_data, threshold)

        # Assert
        np.testing.assert_array_equal(pose_data_v3[:, 0, :, :], pose_data)
        np.testing.assert_array_equal(instance_count, np.ones(2, dtype=np.uint8))
