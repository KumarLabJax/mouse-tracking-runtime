"""Comprehensive unit tests for the multi_to_v2 pose conversion function."""

import numpy as np
import pytest

from mouse_tracking.pose.convert import multi_to_v2


class TestMultiToV2BasicFunctionality:
    """Test basic functionality and successful conversions."""

    def test_single_identity_conversion(self):
        """Test conversion with a single identity across multiple frames."""
        # Arrange
        num_frames, max_animals = 5, 2
        pose_data = np.random.rand(num_frames, max_animals, 12, 2) * 100
        conf_data = (
            np.random.rand(num_frames, max_animals, 12) * 0.8 + 0.2
        )  # 0.2-1.0 range

        # Single identity (ID 1) appears in animal slot 0 for all frames
        identity_data = np.zeros((num_frames, max_animals), dtype=np.uint32)
        identity_data[:, 0] = 1  # Identity 1 in slot 0
        # Slot 1 has all zero confidence (invalid poses)
        conf_data[:, 1, :] = 0.0

        # Act
        result = multi_to_v2(pose_data, conf_data, identity_data)

        # Assert
        assert len(result) == 1  # Only one identity
        identity_id, single_pose, single_conf = result[0]

        assert identity_id == 1
        assert single_pose.shape == (num_frames, 12, 2)
        assert single_conf.shape == (num_frames, 12)
        assert single_pose.dtype == pose_data.dtype
        assert single_conf.dtype == conf_data.dtype

        # Check that pose data from slot 0 is correctly extracted
        np.testing.assert_array_equal(single_pose, pose_data[:, 0, :, :])
        np.testing.assert_array_equal(single_conf, conf_data[:, 0, :])

    def test_multiple_identities_conversion(self):
        """Test conversion with multiple identities."""
        # Arrange
        num_frames = 4
        pose_data = np.ones((num_frames, 3, 12, 2)) * 10
        conf_data = np.ones((num_frames, 3, 12)) * 0.8

        # Set up identities: ID 1 in slot 0, ID 2 in slot 1, slot 2 invalid
        identity_data = np.array(
            [
                [1, 2, 0],  # Frame 0: ID 1 in slot 0, ID 2 in slot 1, slot 2 invalid
                [1, 2, 0],  # Frame 1: same pattern
                [1, 2, 0],  # Frame 2: same pattern
                [1, 2, 0],  # Frame 3: same pattern
            ],
            dtype=np.uint32,
        )

        # Make slot 2 invalid by setting confidence to 0
        conf_data[:, 2, :] = 0.0

        # Act
        result = multi_to_v2(pose_data, conf_data, identity_data)

        # Assert
        assert len(result) == 2  # Two identities

        # Sort results by identity ID for consistent testing
        result.sort(key=lambda x: x[0])

        id1, pose1, conf1 = result[0]
        id2, pose2, conf2 = result[1]

        assert id1 == 1
        assert id2 == 2

        # Check shapes
        for pose, conf in [(pose1, conf1), (pose2, conf2)]:
            assert pose.shape == (num_frames, 12, 2)
            assert conf.shape == (num_frames, 12)

        # Check data extraction
        np.testing.assert_array_equal(pose1, pose_data[:, 0, :, :])  # ID 1 from slot 0
        np.testing.assert_array_equal(conf1, conf_data[:, 0, :])
        np.testing.assert_array_equal(pose2, pose_data[:, 1, :, :])  # ID 2 from slot 1
        np.testing.assert_array_equal(conf2, conf_data[:, 1, :])

    def test_sparse_identity_across_frames(self):
        """Test identity that appears only in some frames."""
        # Arrange
        num_frames = 6
        pose_data = np.ones((num_frames, 2, 12, 2)) * 50
        conf_data = np.ones((num_frames, 2, 12)) * 0.9

        # Identity 1 appears in frames 1, 3, 5 in slot 0
        identity_data = np.zeros((num_frames, 2), dtype=np.uint32)
        identity_frames = [1, 3, 5]
        identity_data[identity_frames, 0] = 1

        # Make other poses invalid
        for frame in range(num_frames):
            if frame not in identity_frames:
                conf_data[frame, 0, :] = 0.0
        conf_data[:, 1, :] = 0.0  # Slot 1 always invalid

        # Act
        result = multi_to_v2(pose_data, conf_data, identity_data)

        # Assert
        assert len(result) == 1
        identity_id, single_pose, single_conf = result[0]

        assert identity_id == 1

        # Check that only identity frames have data, others are zeros
        for frame in range(num_frames):
            if frame in identity_frames:
                np.testing.assert_array_equal(
                    single_pose[frame], pose_data[frame, 0, :, :]
                )
                np.testing.assert_array_equal(
                    single_conf[frame], conf_data[frame, 0, :]
                )
            else:
                np.testing.assert_array_equal(single_pose[frame], np.zeros((12, 2)))
                np.testing.assert_array_equal(single_conf[frame], np.zeros(12))

    def test_identity_switching_slots(self):
        """Test identity that appears in different animal slots across frames."""
        # Arrange
        num_frames = 4
        pose_data = np.arange(num_frames * 3 * 12 * 2).reshape(num_frames, 3, 12, 2)
        conf_data = np.ones((num_frames, 3, 12)) * 0.8

        # Identity 1 switches slots: frame 0 slot 0, frame 1 slot 1, frame 2 slot 2, frame 3 slot 0
        identity_data = np.zeros((num_frames, 3), dtype=np.uint32)
        identity_data[0, 0] = 1  # Frame 0, slot 0
        identity_data[1, 1] = 1  # Frame 1, slot 1
        identity_data[2, 2] = 1  # Frame 2, slot 2
        identity_data[3, 0] = 1  # Frame 3, slot 0

        # Make other slots invalid by setting confidence to 0
        for frame in range(num_frames):
            for slot in range(3):
                if identity_data[frame, slot] != 1:
                    conf_data[frame, slot, :] = 0.0

        # Act
        result = multi_to_v2(pose_data, conf_data, identity_data)

        # Assert
        assert len(result) == 1
        identity_id, single_pose, single_conf = result[0]

        assert identity_id == 1

        # Check that data comes from correct slots
        np.testing.assert_array_equal(
            single_pose[0], pose_data[0, 0, :, :]
        )  # Frame 0, slot 0
        np.testing.assert_array_equal(
            single_pose[1], pose_data[1, 1, :, :]
        )  # Frame 1, slot 1
        np.testing.assert_array_equal(
            single_pose[2], pose_data[2, 2, :, :]
        )  # Frame 2, slot 2
        np.testing.assert_array_equal(
            single_pose[3], pose_data[3, 0, :, :]
        )  # Frame 3, slot 0


class TestMultiToV2EdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_frames(self):
        """Test conversion with zero frames."""
        # Arrange
        pose_data = np.empty((0, 2, 12, 2))
        conf_data = np.empty((0, 2, 12))
        identity_data = np.empty((0, 2), dtype=np.uint32)

        # Act
        result = multi_to_v2(pose_data, conf_data, identity_data)

        # Assert
        assert len(result) == 0  # No identities

    def test_single_frame_single_identity(self):
        """Test conversion with only one frame and one identity."""
        # Arrange
        pose_data = np.ones((1, 2, 12, 2)) * 42
        conf_data = np.ones((1, 2, 12)) * 0.7
        identity_data = np.array([[1, 0]], dtype=np.uint32)
        conf_data[0, 1, :] = 0.0  # Make slot 1 invalid

        # Act
        result = multi_to_v2(pose_data, conf_data, identity_data)

        # Assert
        assert len(result) == 1
        identity_id, single_pose, single_conf = result[0]

        assert identity_id == 1
        assert single_pose.shape == (1, 12, 2)
        assert single_conf.shape == (1, 12)
        np.testing.assert_array_equal(single_pose[0], pose_data[0, 0, :, :])
        np.testing.assert_array_equal(single_conf[0], conf_data[0, 0, :])

    def test_all_invalid_poses(self):
        """Test conversion when all poses are invalid (zero confidence)."""
        # Arrange
        pose_data = np.ones((3, 2, 12, 2)) * 10
        conf_data = np.zeros((3, 2, 12))  # All confidence is zero
        identity_data = np.array([[1, 2], [1, 2], [1, 2]], dtype=np.uint32)

        # Act
        result = multi_to_v2(pose_data, conf_data, identity_data)

        # Assert
        assert len(result) == 0  # No valid identities

    def test_identity_zero_handling(self):
        """Test that identity 0 is properly handled when it has valid poses."""
        # Arrange
        pose_data = np.ones((2, 2, 12, 2)) * 25
        conf_data = np.ones((2, 2, 12)) * 0.8

        # Identity 0 in slot 0, slot 1 invalid
        identity_data = np.array([[0, 0], [0, 0]], dtype=np.uint32)
        conf_data[:, 1, :] = 0.0  # Make slot 1 invalid

        # Act
        result = multi_to_v2(pose_data, conf_data, identity_data)

        # Assert
        assert len(result) == 1
        identity_id, single_pose, single_conf = result[0]

        assert identity_id == 0
        np.testing.assert_array_equal(single_pose, pose_data[:, 0, :, :])
        np.testing.assert_array_equal(single_conf, conf_data[:, 0, :])

    def test_partial_confidence_zero(self):
        """Test poses where only some keypoints have zero confidence."""
        # Arrange
        pose_data = np.ones((2, 2, 12, 2)) * 15
        conf_data = np.ones((2, 2, 12)) * 0.6

        # Set some keypoints to zero confidence but not all
        conf_data[0, 0, :6] = 0.0  # First 6 keypoints zero in frame 0, slot 0
        conf_data[1, 0, 6:] = 0.0  # Last 6 keypoints zero in frame 1, slot 0

        identity_data = np.array([[1, 0], [1, 0]], dtype=np.uint32)
        conf_data[:, 1, :] = 0.0  # Make slot 1 invalid

        # Act
        result = multi_to_v2(pose_data, conf_data, identity_data)

        # Assert
        assert len(result) == 1
        identity_id, single_pose, single_conf = result[0]

        assert identity_id == 1
        # The poses should still be considered valid since not ALL keypoints are zero
        np.testing.assert_array_equal(single_pose, pose_data[:, 0, :, :])
        np.testing.assert_array_equal(single_conf, conf_data[:, 0, :])

    def test_large_identity_numbers(self):
        """Test with large identity numbers."""
        # Arrange
        pose_data = np.ones((2, 2, 12, 2)) * 30
        conf_data = np.ones((2, 2, 12)) * 0.8

        # Use large identity numbers
        identity_data = np.array([[999, 0], [1000, 0]], dtype=np.uint32)
        conf_data[:, 1, :] = 0.0  # Make slot 1 invalid

        # Act
        result = multi_to_v2(pose_data, conf_data, identity_data)

        # Assert
        assert len(result) == 2
        result.sort(key=lambda x: x[0])

        assert result[0][0] == 999
        assert result[1][0] == 1000


class TestMultiToV2ErrorHandling:
    """Test error conditions and invalid inputs."""

    def test_duplicate_identity_same_frame_raises_error(self):
        """Test that duplicate identities in the same frame raise ValueError."""
        # Arrange
        pose_data = np.ones((2, 3, 12, 2)) * 20
        conf_data = np.ones((2, 3, 12)) * 0.8

        # Identity 1 appears in both slot 0 and slot 1 in frame 0
        identity_data = np.array(
            [
                [1, 1, 0],  # Frame 0: ID 1 in both slots 0 and 1 - ERROR!
                [1, 2, 0],  # Frame 1: normal
            ],
            dtype=np.uint32,
        )
        conf_data[:, 2, :] = 0.0  # Make slot 2 invalid

        # Act & Assert
        with pytest.raises(
            ValueError, match="Identity 1 contained multiple poses assigned on frames"
        ):
            multi_to_v2(pose_data, conf_data, identity_data)

    def test_multiple_duplicate_frames_error_message(self):
        """Test error message when identity has duplicates in multiple frames."""
        # Arrange
        pose_data = np.ones((4, 3, 12, 2)) * 20
        conf_data = np.ones((4, 3, 12)) * 0.8

        # Identity 1 appears multiple times in frames 0 and 2
        identity_data = np.array(
            [
                [1, 1, 0],  # Frame 0: ID 1 in both slots 0 and 1
                [1, 2, 0],  # Frame 1: normal
                [1, 1, 0],  # Frame 2: ID 1 in both slots 0 and 1 again
                [1, 2, 0],  # Frame 3: normal
            ],
            dtype=np.uint32,
        )
        conf_data[:, 2, :] = 0.0  # Make slot 2 invalid

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            multi_to_v2(pose_data, conf_data, identity_data)

        error_msg = str(exc_info.value)
        assert "Identity 1" in error_msg
        assert "multiple poses assigned on frames" in error_msg
        # Should mention both frames 0 and 2
        assert "[0 2]" in error_msg

    def test_mismatched_array_shapes(self):
        """Test error handling with mismatched input array shapes."""
        # Arrange
        pose_data = np.ones((5, 2, 12, 2))
        conf_data = np.ones((3, 2, 12))  # Different number of frames
        identity_data = np.ones((5, 2), dtype=np.uint32)

        # Act & Assert
        # This should fail during array operations
        with pytest.raises((IndexError, ValueError)):
            multi_to_v2(pose_data, conf_data, identity_data)

    def test_wrong_pose_data_dimensions(self):
        """Test error handling with incorrect pose data dimensions."""
        # Arrange
        pose_data = np.ones((5, 2, 12))  # Missing coordinate dimension
        conf_data = np.ones((5, 2, 12))
        identity_data = np.ones((5, 2), dtype=np.uint32)

        # Act & Assert
        with pytest.raises((IndexError, ValueError)):
            multi_to_v2(pose_data, conf_data, identity_data)


class TestMultiToV2DataTypes:
    """Test data type handling and preservation."""

    @pytest.mark.parametrize(
        "pose_dtype,conf_dtype",
        [
            (np.float32, np.float32),
            (np.float64, np.float64),
            (np.float32, np.float64),
            (np.float64, np.float32),
            (np.int32, np.float32),
        ],
        ids=[
            "both_float32",
            "both_float64",
            "pose_float32_conf_float64",
            "pose_float64_conf_float32",
            "pose_int32_conf_float32",
        ],
    )
    def test_data_type_preservation(self, pose_dtype, conf_dtype):
        """Test that input data types are preserved in output."""
        # Arrange
        pose_data = np.ones((3, 2, 12, 2), dtype=pose_dtype) * 10
        conf_data = np.ones((3, 2, 12), dtype=conf_dtype) * 0.8
        identity_data = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.uint32)
        conf_data[:, 1, :] = 0.0  # Make slot 1 invalid

        # Act
        result = multi_to_v2(pose_data, conf_data, identity_data)

        # Assert
        assert len(result) == 1
        identity_id, single_pose, single_conf = result[0]

        assert single_pose.dtype == pose_dtype
        assert single_conf.dtype == conf_dtype

    def test_identity_data_type_handling(self):
        """Test handling of different identity data types."""
        # Arrange
        pose_data = np.ones((2, 2, 12, 2)) * 10
        conf_data = np.ones((2, 2, 12)) * 0.8

        # Use different integer types for identity
        identity_data = np.array([[1, 0], [2, 0]], dtype=np.uint16)
        conf_data[:, 1, :] = 0.0

        # Act
        result = multi_to_v2(pose_data, conf_data, identity_data)

        # Assert
        assert len(result) == 2
        result.sort(key=lambda x: x[0])
        assert result[0][0] == 1
        assert result[1][0] == 2


class TestMultiToV2ComplexScenarios:
    """Test complex real-world scenarios."""

    def test_realistic_multi_mouse_tracking(self):
        """Test realistic scenario with multiple mice tracked across frames."""
        # Arrange
        num_frames = 10
        max_animals = 4
        pose_data = np.random.rand(num_frames, max_animals, 12, 2) * 200
        conf_data = np.random.rand(num_frames, max_animals, 12) * 0.8 + 0.2

        # Set up realistic identity tracking pattern
        identity_data = np.zeros((num_frames, max_animals), dtype=np.uint32)

        # Mouse 1: appears in first 6 frames, slot varies
        mouse1_frames = list(range(6))
        mouse1_slots = [0, 0, 1, 1, 2, 2]
        for frame, slot in zip(mouse1_frames, mouse1_slots, strict=False):
            identity_data[frame, slot] = 1

        # Mouse 2: appears in frames 2-8, slot varies
        mouse2_frames = list(range(2, 9))
        mouse2_slots = [2, 3, 0, 3, 0, 1, 3]
        for frame, slot in zip(mouse2_frames, mouse2_slots, strict=False):
            identity_data[frame, slot] = 2

        # Mouse 3: appears sporadically
        mouse3_data = [(1, 3), (4, 1), (7, 0), (9, 2)]
        for frame, slot in mouse3_data:
            identity_data[frame, slot] = 3

        # Set invalid poses (zero confidence for unused slots)
        for frame in range(num_frames):
            for slot in range(max_animals):
                if identity_data[frame, slot] == 0:
                    conf_data[frame, slot, :] = 0.0

        # Act
        result = multi_to_v2(pose_data, conf_data, identity_data)

        # Assert
        assert len(result) == 3  # Three mice
        result.sort(key=lambda x: x[0])

        # Check each mouse
        for i, (mouse_id, single_pose, single_conf) in enumerate(result, 1):
            assert mouse_id == i
            assert single_pose.shape == (num_frames, 12, 2)
            assert single_conf.shape == (num_frames, 12)

            # Verify data extraction for each mouse
            for frame in range(num_frames):
                frame_slots = np.where(identity_data[frame, :] == mouse_id)[0]
                if len(frame_slots) == 1:
                    slot = frame_slots[0]
                    np.testing.assert_array_equal(
                        single_pose[frame], pose_data[frame, slot, :, :]
                    )
                    np.testing.assert_array_equal(
                        single_conf[frame], conf_data[frame, slot, :]
                    )
                else:
                    # No data for this mouse in this frame
                    np.testing.assert_array_equal(single_pose[frame], np.zeros((12, 2)))
                    np.testing.assert_array_equal(single_conf[frame], np.zeros(12))

    def test_identity_appearing_disappearing(self):
        """Test identity that appears, disappears, then reappears."""
        # Arrange
        num_frames = 8
        pose_data = np.ones((num_frames, 2, 12, 2)) * 33
        conf_data = np.ones((num_frames, 2, 12)) * 0.7

        # Identity 1: frames 0-2, then disappears, then reappears frames 5-7
        identity_data = np.zeros((num_frames, 2), dtype=np.uint32)
        appear_frames = [0, 1, 2, 5, 6, 7]
        for frame in appear_frames:
            identity_data[frame, 0] = 1

        # Make slot 1 and frames where identity doesn't appear invalid
        for frame in range(num_frames):
            conf_data[frame, 1, :] = 0.0
            if frame not in appear_frames:
                conf_data[frame, 0, :] = 0.0

        # Act
        result = multi_to_v2(pose_data, conf_data, identity_data)

        # Assert
        assert len(result) == 1
        identity_id, single_pose, single_conf = result[0]

        assert identity_id == 1

        # Check that data appears in correct frames
        for frame in range(num_frames):
            if frame in appear_frames:
                np.testing.assert_array_equal(
                    single_pose[frame], pose_data[frame, 0, :, :]
                )
                np.testing.assert_array_equal(
                    single_conf[frame], conf_data[frame, 0, :]
                )
            else:
                np.testing.assert_array_equal(single_pose[frame], np.zeros((12, 2)))
                np.testing.assert_array_equal(single_conf[frame], np.zeros(12))

    def test_confidence_threshold_boundary(self):
        """Test behavior at confidence threshold boundaries."""
        # Arrange
        pose_data = np.ones((3, 2, 12, 2)) * 40
        conf_data = np.array(
            [
                [
                    [0.0] * 12,
                    [0.1] * 12,
                ],  # Frame 0: slot 0 all zero (invalid), slot 1 low conf (valid)
                [
                    [0.0001] * 12,
                    [0.0] * 12,
                ],  # Frame 1: slot 0 very low conf (valid), slot 1 zero (invalid)
                [
                    [0.5] * 12,
                    [0.0] * 12,
                ],  # Frame 2: slot 0 medium conf (valid), slot 1 zero (invalid)
            ]
        )

        identity_data = np.array(
            [
                [1, 2],  # Frame 0
                [1, 2],  # Frame 1
                [1, 2],  # Frame 2
            ],
            dtype=np.uint32,
        )

        # Act
        result = multi_to_v2(pose_data, conf_data, identity_data)

        # Assert
        # Both identities should appear:
        # - Identity 1 has valid poses in frames 1,2 (frame 0 slot 0 is all zero)
        # - Identity 2 has valid pose in frame 0 (frames 1,2 slot 1 are all zero)
        assert len(result) == 2
        result.sort(key=lambda x: x[0])

        identity1_id, pose1, conf1 = result[0]
        identity2_id, pose2, conf2 = result[1]

        assert identity1_id == 1
        assert identity2_id == 2

        # Identity 1: Frame 0 should be zeros, frames 1,2 should have data
        np.testing.assert_array_equal(pose1[0], np.zeros((12, 2)))
        np.testing.assert_array_equal(conf1[0], np.zeros(12))
        np.testing.assert_array_equal(pose1[1], pose_data[1, 0, :, :])
        np.testing.assert_array_equal(conf1[1], conf_data[1, 0, :])
        np.testing.assert_array_equal(pose1[2], pose_data[2, 0, :, :])
        np.testing.assert_array_equal(conf1[2], conf_data[2, 0, :])

        # Identity 2: Frame 0 should have data, frames 1,2 should be zeros
        np.testing.assert_array_equal(pose2[0], pose_data[0, 1, :, :])
        np.testing.assert_array_equal(conf2[0], conf_data[0, 1, :])
        np.testing.assert_array_equal(pose2[1], np.zeros((12, 2)))
        np.testing.assert_array_equal(conf2[1], np.zeros(12))
        np.testing.assert_array_equal(pose2[2], np.zeros((12, 2)))
        np.testing.assert_array_equal(conf2[2], np.zeros(12))

    @pytest.mark.parametrize(
        "max_animals",
        [1, 2, 4, 8],
        ids=["single_animal", "two_animals", "four_animals", "eight_animals"],
    )
    def test_different_max_animals(self, max_animals):
        """Test function with different maximum animal counts."""
        # Arrange
        num_frames = 3
        pose_data = np.ones((num_frames, max_animals, 12, 2)) * 60
        conf_data = np.ones((num_frames, max_animals, 12)) * 0.8

        # Create identities 1 to max_animals in corresponding slots
        identity_data = np.zeros((num_frames, max_animals), dtype=np.uint32)
        for slot in range(max_animals):
            identity_data[:, slot] = slot + 1  # IDs 1, 2, 3, ...

        # Act
        result = multi_to_v2(pose_data, conf_data, identity_data)

        # Assert
        assert len(result) == max_animals
        result.sort(key=lambda x: x[0])

        for i, (identity_id, single_pose, single_conf) in enumerate(result):
            assert identity_id == i + 1
            assert single_pose.shape == (num_frames, 12, 2)
            assert single_conf.shape == (num_frames, 12)
            np.testing.assert_array_equal(single_pose, pose_data[:, i, :, :])
            np.testing.assert_array_equal(single_conf, conf_data[:, i, :])

    def test_large_dataset_performance(self):
        """Test function performance with large datasets."""
        # Arrange
        num_frames = 1000
        max_animals = 5
        pose_data = (
            np.random.rand(num_frames, max_animals, 12, 2).astype(np.float32) * 100
        )
        conf_data = (
            np.random.rand(num_frames, max_animals, 12).astype(np.float32) * 0.8 + 0.2
        )

        # Create sparse identity pattern for performance testing
        identity_data = np.zeros((num_frames, max_animals), dtype=np.uint32)

        # Identity 1: every 5th frame starting from 0
        identity_data[::5, 0] = 1
        # Identity 2: every 7th frame starting from 1
        identity_data[1::7, 1] = 2
        # Identity 3: every 10th frame starting from 2
        identity_data[2::10, 2] = 3

        # Set invalid poses for unused slots
        for frame in range(num_frames):
            for slot in range(max_animals):
                if identity_data[frame, slot] == 0:
                    conf_data[frame, slot, :] = 0.0

        # Act (should complete without performance issues)
        result = multi_to_v2(pose_data, conf_data, identity_data)

        # Assert
        assert len(result) == 3  # Three identities
        result.sort(key=lambda x: x[0])

        for _identity_id, single_pose, single_conf in result:
            assert single_pose.shape == (num_frames, 12, 2)
            assert single_conf.shape == (num_frames, 12)
            assert single_pose.dtype == np.float32
            assert single_conf.dtype == np.float32
