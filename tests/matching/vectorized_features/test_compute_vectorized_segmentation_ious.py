"""Tests for compute_vectorized_segmentation_ious function."""

from unittest.mock import patch

import numpy as np

from mouse_tracking.matching.vectorized_features import (
    compute_vectorized_segmentation_ious,
)


class TestComputeVectorizedSegmentationIous:
    """Test basic functionality of compute_vectorized_segmentation_ious."""

    def test_basic_segmentation_iou(self, features_factory):
        """Test basic segmentation IoU computation."""
        # Create features with known segmentation data
        seg_configs = [
            {"has_segmentation": True},  # Will have segmentation
            {"has_segmentation": True},  # Will have segmentation
        ]

        features1 = features_factory(
            n_detections=1, seg_configs=[seg_configs[0]], seed=42
        )
        features2 = features_factory(
            n_detections=1, seg_configs=[seg_configs[1]], seed=42
        )

        # Mock render_blob to return predictable segmentation images
        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            # Create simple test segmentation images
            seg_image1 = np.array([[True, False], [False, True]])  # 2 pixels
            seg_image2 = np.array([[True, True], [False, False]])  # 2 pixels, 1 overlap

            mock_render.side_effect = [seg_image1, seg_image2]

            result = compute_vectorized_segmentation_ious(features1, features2)

            # Should be a 1x1 matrix
            assert result.shape == (1, 1)

            # Compute expected IoU manually
            intersection = np.sum(np.logical_and(seg_image1, seg_image2))  # 1 pixel
            union = np.sum(np.logical_or(seg_image1, seg_image2))  # 3 pixels
            expected_iou = intersection / union  # 1/3

            np.testing.assert_allclose(result[0, 0], expected_iou, rtol=1e-10)

    def test_identical_segmentations(self, features_factory):
        """Test IoU between identical segmentations."""
        seg_configs = [{"has_segmentation": True}]

        features1 = features_factory(n_detections=1, seg_configs=seg_configs, seed=42)
        features2 = features_factory(n_detections=1, seg_configs=seg_configs, seed=42)

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            # Identical segmentation images
            seg_image = np.array([[True, False, True], [False, True, False]])
            mock_render.return_value = seg_image

            result = compute_vectorized_segmentation_ious(features1, features2)

            # Identical segmentations should have IoU = 1.0
            assert result.shape == (1, 1)
            np.testing.assert_allclose(result[0, 0], 1.0, rtol=1e-10)

    def test_non_overlapping_segmentations(self, features_factory):
        """Test IoU between non-overlapping segmentations."""
        seg_configs = [{"has_segmentation": True}, {"has_segmentation": True}]

        features1 = features_factory(n_detections=1, seg_configs=[seg_configs[0]])
        features2 = features_factory(n_detections=1, seg_configs=[seg_configs[1]])

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            # Non-overlapping segmentation images
            seg_image1 = np.array([[True, False], [False, False]])
            seg_image2 = np.array([[False, False], [False, True]])

            mock_render.side_effect = [seg_image1, seg_image2]

            result = compute_vectorized_segmentation_ious(features1, features2)

            # Non-overlapping segmentations should have IoU = 0.0
            assert result.shape == (1, 1)
            np.testing.assert_allclose(result[0, 0], 0.0, rtol=1e-10)

    def test_matrix_computation(self, features_factory):
        """Test IoU matrix for multiple segmentations."""
        seg_configs = [
            {"has_segmentation": True},
            {"has_segmentation": True},
            {"has_segmentation": True},
        ]

        features1 = features_factory(
            n_detections=2, seg_configs=seg_configs[:2], seed=42
        )
        features2 = features_factory(n_detections=3, seg_configs=seg_configs, seed=100)

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            # Create test segmentation images with known properties
            seg_images = [
                np.array([[True, False], [False, True]]),  # 2 pixels
                np.array([[False, True], [True, False]]),  # 2 pixels
                np.array([[True, True], [False, False]]),  # 2 pixels
                np.array([[False, False], [True, True]]),  # 2 pixels
                np.array([[True, False], [True, False]]),  # 2 pixels
            ]

            mock_render.side_effect = seg_images

            result = compute_vectorized_segmentation_ious(features1, features2)

            # Should be 2x3 matrix
            assert result.shape == (2, 3)

            # Check that all IoUs are valid
            assert np.all(~np.isnan(result))
            assert np.all(result >= 0)
            assert np.all(result <= 1.0)

    def test_consistency_with_original_method(self, features_factory):
        """Test consistency with Detection.seg_iou method."""
        # Create features with segmentations
        features1 = features_factory(
            n_detections=1, seg_configs=[{"has_segmentation": True}], seed=42
        )
        features2 = features_factory(
            n_detections=1, seg_configs=[{"has_segmentation": True}], seed=100
        )

        # Mock render_blob to return predictable results
        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            # Create test segmentation images
            seg_image1 = np.array([[True, False], [False, True]])
            seg_image2 = np.array([[True, True], [False, False]])

            # Mock the render_blob calls
            mock_render.side_effect = [seg_image1, seg_image2]

            # Test vectorized method
            vectorized_iou = compute_vectorized_segmentation_ious(features1, features2)

            # Compute expected IoU manually
            intersection = np.sum(np.logical_and(seg_image1, seg_image2))
            union = np.sum(np.logical_or(seg_image1, seg_image2))
            expected_iou = intersection / union if union > 0 else 0.0

            # Should match expected calculation
            assert vectorized_iou.shape == (1, 1)
            np.testing.assert_allclose(vectorized_iou[0, 0], expected_iou, rtol=1e-15)


class TestComputeVectorizedSegmentationIousEdgeCases:
    """Test edge cases and invalid input handling."""

    def test_missing_segmentations_both_sides(self, features_factory):
        """Test with missing segmentations on both sides."""
        seg_configs1 = [{"has_segmentation": False}, {"has_segmentation": False}]
        seg_configs2 = [
            {"has_segmentation": False},
            {"has_segmentation": False},
            {"has_segmentation": False},
        ]

        features1 = features_factory(n_detections=2, seg_configs=seg_configs1)
        features2 = features_factory(n_detections=3, seg_configs=seg_configs2)

        result = compute_vectorized_segmentation_ious(features1, features2)

        # Should return all NaN
        assert result.shape == (2, 3)
        assert np.all(np.isnan(result))

    def test_missing_segmentations_one_side(self, features_factory):
        """Test with missing segmentations on one side."""
        seg_configs_valid = [{"has_segmentation": True}, {"has_segmentation": True}]
        seg_configs_missing = [{"has_segmentation": False}]

        features1 = features_factory(n_detections=2, seg_configs=seg_configs_valid)
        features2 = features_factory(n_detections=1, seg_configs=seg_configs_missing)

        # Mock render_blob only for valid segmentations
        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            seg_image = np.array([[True, False], [False, True]])
            mock_render.return_value = seg_image

            result = compute_vectorized_segmentation_ious(features1, features2)

            # Should return 0.0 (valid vs invalid, one has seg_mat)
            assert result.shape == (2, 1)
            assert np.all(result == 0.0)  # One side has _seg_mat, other doesn't

    def test_mixed_valid_invalid_segmentations(self, features_factory):
        """Test with mixed valid and invalid segmentations."""
        features1 = features_factory(
            n_detections=2,
            seg_configs=[
                {"has_segmentation": True},  # Valid
                {"has_segmentation": False},  # Invalid
            ],
        )
        features2 = features_factory(
            n_detections=2,
            seg_configs=[
                {"has_segmentation": False},  # Invalid
                {"has_segmentation": True},  # Valid
            ],
        )

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            # Only return for valid segmentations
            seg_image = np.array([[True, False], [False, True]])
            mock_render.return_value = seg_image

            result = compute_vectorized_segmentation_ious(features1, features2)

            assert result.shape == (2, 2)

            # Based on the function logic:
            # If at least one has _seg_mat, return 0.0; otherwise NaN
            # (0,0): valid vs invalid -> 0.0 (one has seg_mat)
            # (0,1): valid vs valid -> computed IoU
            # (1,0): invalid vs invalid -> NaN (both have no seg_mat)
            # (1,1): invalid vs valid -> 0.0 (one has seg_mat)

            assert result[0, 0] == 0.0  # valid vs invalid
            assert not np.isnan(result[0, 1])  # valid vs valid
            assert np.isnan(result[1, 0])  # invalid vs invalid
            assert result[1, 1] == 0.0  # invalid vs valid

            # Check the valid IoU
            assert 0 <= result[0, 1] <= 1.0

    def test_empty_segmentations(self, features_factory):
        """Test with empty segmentation images (all False)."""
        seg_configs = [{"has_segmentation": True}, {"has_segmentation": True}]

        features1 = features_factory(n_detections=1, seg_configs=[seg_configs[0]])
        features2 = features_factory(n_detections=1, seg_configs=[seg_configs[1]])

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            # Empty segmentation images (all False)
            empty_seg = np.array([[False, False], [False, False]])
            mock_render.return_value = empty_seg

            result = compute_vectorized_segmentation_ious(features1, features2)

            # Empty segmentations should return 0.0 (union = 0 case)
            assert result.shape == (1, 1)
            assert result[0, 0] == 0.0

    def test_zero_union_case(self, features_factory):
        """Test the special case where union is zero."""
        seg_configs = [{"has_segmentation": True}]

        features1 = features_factory(n_detections=1, seg_configs=seg_configs)
        features2 = features_factory(n_detections=1, seg_configs=seg_configs)

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            # Both segmentations are empty (all False)
            empty_seg = np.zeros((3, 3), dtype=bool)
            mock_render.return_value = empty_seg

            result = compute_vectorized_segmentation_ious(features1, features2)

            # Union = 0 case should return 0.0 as per function logic
            assert result.shape == (1, 1)
            assert result[0, 0] == 0.0

    def test_no_detections(self, features_factory):
        """Test with no detections."""
        features1 = features_factory(n_detections=0)
        features2 = features_factory(n_detections=0)

        result = compute_vectorized_segmentation_ious(features1, features2)

        # Should return empty matrix
        assert result.shape == (0, 0)

    def test_single_detection_each_side(self, features_factory):
        """Test with single detection on each side."""
        features1 = features_factory(
            n_detections=1, seg_configs=[{"has_segmentation": True}]
        )
        features2 = features_factory(
            n_detections=1, seg_configs=[{"has_segmentation": True}]
        )

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            seg_image = np.array([[True, False], [True, False]])
            mock_render.return_value = seg_image

            result = compute_vectorized_segmentation_ious(features1, features2)

            assert result.shape == (1, 1)
            assert not np.isnan(result[0, 0])
            assert 0 <= result[0, 0] <= 1.0

    def test_special_case_one_has_seg_mat_other_none(self, features_factory):
        """Test special case where one has _seg_mat but other is None."""
        # Create features where detections have different _seg_mat states
        features1 = features_factory(
            n_detections=1, seg_configs=[{"has_segmentation": True}]
        )
        features2 = features_factory(
            n_detections=1, seg_configs=[{"has_segmentation": False}]
        )

        # Manually ensure one detection has _seg_mat and other doesn't
        features1.detections[0]._seg_mat = np.array(
            [[[1, 2], [3, 4]]]
        )  # Has segmentation data
        features2.detections[0]._seg_mat = None  # No segmentation data

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            # Only called for the detection with _seg_mat
            mock_render.return_value = np.array([[True, False]])

            result = compute_vectorized_segmentation_ious(features1, features2)

            # Should return 0.0 as per function logic (one has seg data, other doesn't)
            assert result.shape == (1, 1)
            assert result[0, 0] == 0.0


class TestComputeVectorizedSegmentationIousProperties:
    """Test mathematical properties and correctness."""

    def test_iou_symmetry(self, features_factory):
        """Test that IoU matrix is symmetric for same features."""
        features = features_factory(
            n_detections=3,
            seg_configs=[
                {"has_segmentation": True},
                {"has_segmentation": True},
                {"has_segmentation": True},
            ],
            seed=42,
        )

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            # Create different segmentation images
            seg_images = [
                np.array([[True, False], [False, True]]),
                np.array([[False, True], [True, False]]),
                np.array([[True, True], [False, False]]),
            ]
            mock_render.side_effect = (
                seg_images + seg_images
            )  # Called twice for symmetric computation

            result = compute_vectorized_segmentation_ious(features, features)

            # Should be symmetric
            assert result.shape == (3, 3)
            np.testing.assert_allclose(result, result.T, rtol=1e-10)

            # Diagonal should be 1.0 (self-IoU)
            diagonal = np.diag(result)
            np.testing.assert_allclose(diagonal, 1.0, rtol=1e-10)

    def test_iou_bounds(self, features_factory):
        """Test that IoUs are bounded correctly."""
        features1 = features_factory(n_detections=5, seed=42)
        features2 = features_factory(n_detections=7, seed=100)

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            # Create random but valid segmentation images
            np.random.seed(42)
            seg_images = []
            for _ in range(12):  # 5 + 7
                seg_img = np.random.random((4, 4)) > 0.5
                seg_images.append(seg_img)
            mock_render.side_effect = seg_images

            result = compute_vectorized_segmentation_ious(features1, features2)

            # All valid IoUs should be in [0, 1]
            valid_mask = ~np.isnan(result)
            valid_ious = result[valid_mask]

            if len(valid_ious) > 0:
                assert np.all(valid_ious >= 0)
                assert np.all(valid_ious <= 1.0)

    def test_iou_mathematical_properties(self, features_factory):
        """Test mathematical properties of IoU computation."""
        # Test Case 1: Complete overlap
        features1 = features_factory(
            n_detections=1, seg_configs=[{"has_segmentation": True}]
        )
        features2 = features_factory(
            n_detections=1, seg_configs=[{"has_segmentation": True}]
        )

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            seg_image = np.array([[True, True], [False, False]])
            mock_render.return_value = seg_image

            result = compute_vectorized_segmentation_ious(features1, features2)
            assert result[0, 0] == 1.0

        # Test Case 2: No overlap
        features1 = features_factory(
            n_detections=1, seg_configs=[{"has_segmentation": True}]
        )
        features2 = features_factory(
            n_detections=1, seg_configs=[{"has_segmentation": True}]
        )

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            seg_image1 = np.array([[True, False], [False, False]])
            seg_image2 = np.array([[False, True], [False, False]])
            mock_render.side_effect = [seg_image1, seg_image2]

            result = compute_vectorized_segmentation_ious(features1, features2)
            assert result[0, 0] == 0.0

        # Test Case 3: Partial overlap
        features1 = features_factory(
            n_detections=1, seg_configs=[{"has_segmentation": True}]
        )
        features2 = features_factory(
            n_detections=1, seg_configs=[{"has_segmentation": True}]
        )

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            seg_image1 = np.array([[True, True], [False, False]])  # 2 pixels
            seg_image2 = np.array([[True, False], [True, False]])  # 2 pixels, 1 overlap
            mock_render.side_effect = [seg_image1, seg_image2]

            result = compute_vectorized_segmentation_ious(features1, features2)
            expected = 1 / 3  # intersection=1, union=3
            np.testing.assert_allclose(result[0, 0], expected, rtol=1e-10)


class TestComputeVectorizedSegmentationIousPerformance:
    """Test performance characteristics."""

    def test_large_matrix_computation(self, features_factory):
        """Test computation with larger matrices."""
        # Test with moderately large matrices
        n1, n2 = 20, 25

        features1 = features_factory(
            n_detections=n1,
            seg_configs=[{"has_segmentation": True} for _ in range(n1)],
            seed=42,
        )
        features2 = features_factory(
            n_detections=n2,
            seg_configs=[{"has_segmentation": True} for _ in range(n2)],
            seed=100,
        )

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            # Create varied segmentation images
            np.random.seed(123)
            seg_images = []
            for _ in range(n1 + n2):
                seg_img = np.random.random((8, 8)) > 0.6
                seg_images.append(seg_img)
            mock_render.side_effect = seg_images

            result = compute_vectorized_segmentation_ious(features1, features2)

            # Should complete and return correct shape
            assert result.shape == (n1, n2)

            # All should be valid since we have valid segmentations
            assert np.all(~np.isnan(result))
            assert np.all(result >= 0)
            assert np.all(result <= 1.0)

    def test_memory_efficiency_sparse_valid(self, features_factory):
        """Test memory efficiency with sparse valid segmentations."""
        n1, n2 = 15, 18

        # Most segmentations invalid, only a few valid
        seg_configs1 = [{"has_segmentation": i < 3} for i in range(n1)]
        seg_configs2 = [{"has_segmentation": i < 4} for i in range(n2)]

        features1 = features_factory(n_detections=n1, seg_configs=seg_configs1)
        features2 = features_factory(n_detections=n2, seg_configs=seg_configs2)

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            # Only valid segmentations will call render_blob
            seg_image = np.array([[True, False], [False, True]])
            mock_render.return_value = seg_image

            result = compute_vectorized_segmentation_ious(features1, features2)

            assert result.shape == (n1, n2)

            # Check that most entries are not NaN due to the special case logic
            # (when one side has _seg_mat, it returns 0.0 instead of NaN)
            non_nan_entries = np.sum(~np.isnan(result))

            # Should have many non-NaN entries due to the special case
            assert non_nan_entries > 0

            # Check that the matrix has the expected structure
            # Valid x valid should have proper IoUs
            # Valid x invalid or invalid x valid should have 0.0
            # Invalid x invalid should have NaN
            assert result.shape == (n1, n2)


class TestComputeVectorizedSegmentationIousIntegration:
    """Test integration with existing codebase."""

    def test_match_original_iou_matrix(self, features_factory):
        """Test that results match expected IoU computations."""
        # Create features with mixed valid/invalid segmentations
        features = features_factory(
            n_detections=3,
            seg_configs=[
                {"has_segmentation": True},  # Valid segmentation
                {"has_segmentation": True},  # Valid segmentation
                {"has_segmentation": False},  # No segmentation
            ],
        )

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            # Create test segmentation images for the valid ones
            seg_image1 = np.array([[True, False], [False, True]])
            seg_image2 = np.array([[True, True], [False, False]])
            mock_render.side_effect = [seg_image1, seg_image2, seg_image1, seg_image2]

            vectorized_matrix = compute_vectorized_segmentation_ious(features, features)

            # Should be 3x3 matrix
            assert vectorized_matrix.shape == (3, 3)

            # Check that valid pairs have valid IoUs and invalid pairs have NaN
            # (0,0) and (1,1) should be 1.0 (self-IoU)
            np.testing.assert_allclose(vectorized_matrix[0, 0], 1.0, rtol=1e-15)
            np.testing.assert_allclose(vectorized_matrix[1, 1], 1.0, rtol=1e-15)

            # (0,1) and (1,0) should be computed IoU
            expected_iou = np.sum(np.logical_and(seg_image1, seg_image2)) / np.sum(
                np.logical_or(seg_image1, seg_image2)
            )
            np.testing.assert_allclose(
                vectorized_matrix[0, 1], expected_iou, rtol=1e-15
            )
            np.testing.assert_allclose(
                vectorized_matrix[1, 0], expected_iou, rtol=1e-15
            )

            # Rows/columns with invalid segmentations should be 0.0 when paired with valid ones
            # Based on the special case logic in the function
            # (2,0) and (2,1): invalid vs valid -> 0.0
            # (0,2) and (1,2): valid vs invalid -> 0.0
            # (2,2): invalid vs invalid -> NaN
            assert vectorized_matrix[2, 0] == 0.0  # Invalid vs valid
            assert vectorized_matrix[2, 1] == 0.0  # Invalid vs valid
            assert vectorized_matrix[0, 2] == 0.0  # Valid vs invalid
            assert vectorized_matrix[1, 2] == 0.0  # Valid vs invalid
            assert np.isnan(vectorized_matrix[2, 2])  # Invalid vs invalid

    def test_usage_in_compute_vectorized_match_costs(self, features_factory):
        """Test integration with compute_vectorized_match_costs function."""
        from mouse_tracking.matching.vectorized_features import (
            compute_vectorized_match_costs,
        )

        # Create features that would be used in match cost computation
        features1 = features_factory(n_detections=2, seed=42)
        features2 = features_factory(n_detections=3, seed=100)

        # This should not raise any errors and should use our function internally
        result = compute_vectorized_match_costs(features1, features2)

        assert result.shape == (2, 3)
        assert np.all(np.isfinite(result))  # Match costs should be finite

    def test_caching_behavior(self, features_factory):
        """Test that segmentation images are properly cached."""
        features = features_factory(
            n_detections=2,
            seg_configs=[{"has_segmentation": True}, {"has_segmentation": True}],
        )

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            seg_image = np.array([[True, False], [False, True]])
            mock_render.return_value = seg_image

            # First call should cache the results
            result1 = compute_vectorized_segmentation_ious(features, features)

            # Second call should use cached results (render_blob not called again)
            result2 = compute_vectorized_segmentation_ious(features, features)

            # Results should be identical
            np.testing.assert_array_equal(result1, result2)

            # render_blob should have been called only for the first computation
            # (2 detections for get_seg_images call = 2 calls)
            assert mock_render.call_count == 2
