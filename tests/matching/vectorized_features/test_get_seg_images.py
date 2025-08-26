"""Tests for VectorizedDetectionFeatures.get_seg_images method."""

from unittest.mock import patch

import numpy as np

from mouse_tracking.matching.vectorized_features import VectorizedDetectionFeatures


class TestGetSegImages:
    """Test get_seg_images method."""

    def test_get_seg_images_basic(self, detection_factory):
        """Test basic segmentation image functionality."""
        detections = [
            detection_factory(pose_idx=0, has_segmentation=True),
            detection_factory(pose_idx=1, has_segmentation=True),
        ]

        features = VectorizedDetectionFeatures(detections)

        # Mock the render_blob function
        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            # Set up mock return values
            mock_render.side_effect = [
                np.ones((100, 100), dtype=bool),  # Mock seg image for first detection
                np.zeros((100, 100), dtype=bool),  # Mock seg image for second detection
            ]

            seg_images = features.get_seg_images()

            # Check that render_blob was called correctly
            assert mock_render.call_count == 2

            # Check the results
            assert len(seg_images) == 2
            assert isinstance(seg_images[0], np.ndarray)
            assert isinstance(seg_images[1], np.ndarray)
            assert seg_images[0].shape == (100, 100)
            assert seg_images[1].shape == (100, 100)
            assert seg_images[0].dtype == bool
            assert seg_images[1].dtype == bool

            # Check that the cached result is stored
            assert features._seg_images is seg_images

    def test_get_seg_images_caching(self, detection_factory):
        """Test that segmentation images are cached."""
        detections = [detection_factory(pose_idx=0, has_segmentation=True)]
        features = VectorizedDetectionFeatures(detections)

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            mock_render.return_value = np.ones((50, 50), dtype=bool)

            # First call should compute
            seg_images1 = features.get_seg_images()
            assert mock_render.call_count == 1

            # Second call should use cache
            seg_images2 = features.get_seg_images()
            assert mock_render.call_count == 1  # Should not be called again

            # Should return the same object
            assert seg_images1 is seg_images2

    def test_get_seg_images_none_segmentation(self, detection_factory):
        """Test handling of None segmentation data."""
        detections = [
            detection_factory(pose_idx=0, has_segmentation=True),
            detection_factory(pose_idx=1, has_segmentation=False),  # No segmentation
        ]

        features = VectorizedDetectionFeatures(detections)

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            mock_render.return_value = np.ones((50, 50), dtype=bool)

            seg_images = features.get_seg_images()

            # Should only call render_blob for the detection with segmentation
            assert mock_render.call_count == 1

            # Check the results
            assert len(seg_images) == 2
            assert isinstance(seg_images[0], np.ndarray)
            assert seg_images[1] is None  # No segmentation

    def test_get_seg_images_all_none(self, detection_factory):
        """Test handling when all segmentations are None."""
        detections = [
            detection_factory(pose_idx=0, has_segmentation=False),
            detection_factory(pose_idx=1, has_segmentation=False),
        ]

        features = VectorizedDetectionFeatures(detections)

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            seg_images = features.get_seg_images()

            # Should not call render_blob at all
            assert mock_render.call_count == 0

            # All should be None
            assert len(seg_images) == 2
            assert seg_images[0] is None
            assert seg_images[1] is None

    def test_get_seg_images_empty_detections(self):
        """Test handling of empty detections list."""
        features = VectorizedDetectionFeatures([])

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            seg_images = features.get_seg_images()

            # Should not call render_blob
            assert mock_render.call_count == 0

            # Should return empty list
            assert len(seg_images) == 0

    def test_get_seg_images_uses_render_blob_correctly(self, detection_factory):
        """Test that the method uses render_blob correctly."""
        detections = [detection_factory(pose_idx=0, has_segmentation=True)]
        features = VectorizedDetectionFeatures(detections)

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            mock_render.return_value = np.ones((75, 75), dtype=bool)

            seg_images = features.get_seg_images()

            # Check that render_blob was called with correct arguments
            assert mock_render.call_count == 1
            call_args = mock_render.call_args

            # First argument should be the segmentation matrix
            seg_mat_arg = call_args[0][0]
            assert seg_mat_arg is not None
            assert seg_mat_arg.shape == (100, 100, 2)  # Default seg_shape from conftest

            # Result should use the mocked return value
            assert isinstance(seg_images[0], np.ndarray)
            assert seg_images[0].shape == (75, 75)

    def test_get_seg_images_mixed_valid_invalid(self, detection_factory):
        """Test with mixed valid and invalid segmentations."""
        detections = [
            detection_factory(pose_idx=0, has_segmentation=True),
            detection_factory(pose_idx=1, has_segmentation=False),
            detection_factory(pose_idx=2, has_segmentation=True),
            detection_factory(pose_idx=3, has_segmentation=False),
        ]

        features = VectorizedDetectionFeatures(detections)

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            mock_render.side_effect = [
                np.ones((60, 60), dtype=bool),  # For detection 0
                np.zeros((60, 60), dtype=bool),  # For detection 2
            ]

            seg_images = features.get_seg_images()

            # Should call render_blob twice (for detections 0 and 2)
            assert mock_render.call_count == 2

            # Check the results
            assert len(seg_images) == 4
            assert isinstance(seg_images[0], np.ndarray)  # Valid
            assert seg_images[1] is None  # Invalid
            assert isinstance(seg_images[2], np.ndarray)  # Valid
            assert seg_images[3] is None  # Invalid

    def test_get_seg_images_access_seg_mat(self, mock_detection):
        """Test that the method correctly accesses _seg_mat attribute."""
        # Create detections with different _seg_mat values
        det1 = mock_detection(pose_idx=0, seg_mat=np.ones((50, 50, 2), dtype=np.int32))
        det2 = mock_detection(pose_idx=1, seg_mat=None)

        detections = [det1, det2]
        features = VectorizedDetectionFeatures(detections)

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            mock_render.return_value = np.ones((25, 25), dtype=bool)

            features.get_seg_images()

            # Should only call render_blob for detection with _seg_mat
            assert mock_render.call_count == 1

            # Check that it was called with the correct _seg_mat
            call_args = mock_render.call_args
            seg_mat_arg = call_args[0][0]
            assert np.array_equal(seg_mat_arg, det1._seg_mat)

    def test_get_seg_images_preserves_original_data(self, detection_factory):
        """Test that original detection data is not modified."""
        detections = [detection_factory(pose_idx=0, has_segmentation=True)]
        features = VectorizedDetectionFeatures(detections)

        # Store original segmentation data
        original_seg_mat = detections[0]._seg_mat.copy()

        with patch(
            "mouse_tracking.matching.vectorized_features.render_blob"
        ) as mock_render:
            mock_render.return_value = np.ones((80, 80), dtype=bool)

            seg_images = features.get_seg_images()

            # Original segmentation data should be unchanged
            assert np.array_equal(detections[0]._seg_mat, original_seg_mat)

            # Rendered image should be different
            assert not np.array_equal(seg_images[0], original_seg_mat)


class TestGetSegImagesIntegration:
    """Integration tests for get_seg_images method."""

    def test_get_seg_images_real_rendering(self, detection_factory):
        """Test with real render_blob (no mocking)."""
        detections = [detection_factory(pose_idx=0, has_segmentation=True)]
        features = VectorizedDetectionFeatures(detections)

        # Get segmentation images (this will use the actual render_blob function)
        seg_images = features.get_seg_images()

        # Check that we got a result
        assert len(seg_images) == 1
        assert isinstance(seg_images[0], np.ndarray)
        assert seg_images[0].dtype == bool

        # Should be a reasonable size (render_blob default is 800x800)
        assert seg_images[0].shape == (800, 800)

    def test_get_seg_images_consistency(self, detection_factory):
        """Test that method produces consistent results."""
        detections = [
            detection_factory(pose_idx=0, has_segmentation=True),
            detection_factory(pose_idx=1, has_segmentation=True),
        ]

        features = VectorizedDetectionFeatures(detections)

        # Get segmentation images multiple times
        seg_images1 = features.get_seg_images()
        seg_images2 = features.get_seg_images()
        seg_images3 = features.get_seg_images()

        # All should be identical (due to caching)
        assert len(seg_images1) == len(seg_images2) == len(seg_images3)
        assert seg_images1 is seg_images2  # Same object due to caching
        assert seg_images2 is seg_images3  # Same object due to caching

        # Individual images should be identical
        for i in range(len(seg_images1)):
            if seg_images1[i] is not None:
                assert np.array_equal(seg_images1[i], seg_images2[i])
                assert np.array_equal(seg_images2[i], seg_images3[i])

    def test_get_seg_images_with_none_segmentation_real(self, detection_factory):
        """Test with real data including None segmentations."""
        detections = [
            detection_factory(pose_idx=0, has_segmentation=True),
            detection_factory(pose_idx=1, has_segmentation=False),
            detection_factory(pose_idx=2, has_segmentation=True),
        ]

        features = VectorizedDetectionFeatures(detections)

        seg_images = features.get_seg_images()

        # Check the results
        assert len(seg_images) == 3
        assert isinstance(seg_images[0], np.ndarray)
        assert seg_images[1] is None
        assert isinstance(seg_images[2], np.ndarray)

        # Valid images should have correct properties
        assert seg_images[0].dtype == bool
        assert seg_images[2].dtype == bool
        assert seg_images[0].shape == (800, 800)
        assert seg_images[2].shape == (800, 800)

    def test_get_seg_images_data_types(self, detection_factory):
        """Test that data types are correct."""
        detections = [detection_factory(pose_idx=0, has_segmentation=True)]
        features = VectorizedDetectionFeatures(detections)

        seg_images = features.get_seg_images()

        # Should be a list
        assert isinstance(seg_images, list)

        # Valid images should be boolean numpy arrays
        assert isinstance(seg_images[0], np.ndarray)
        assert seg_images[0].dtype == bool

    def test_get_seg_images_empty_real(self):
        """Test with empty detections using real render_blob."""
        features = VectorizedDetectionFeatures([])

        seg_images = features.get_seg_images()

        # Should return empty list
        assert isinstance(seg_images, list)
        assert len(seg_images) == 0
