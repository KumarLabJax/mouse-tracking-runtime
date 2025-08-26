"""Shared fixtures and utilities for vectorized features testing."""

from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def mock_detection():
    """Create a factory function for mock Detection objects."""

    def _create_mock_detection(
        frame: int = 0,
        pose_idx: int = 0,
        pose: np.ndarray = None,
        embed: np.ndarray = None,
        seg_idx: int = 0,
        seg_mat: np.ndarray = None,
        seg_img: np.ndarray = None,
    ):
        """Create a mock Detection object with specified attributes.

        Args:
            frame: Frame index
            pose_idx: Pose index in frame
            pose: Pose data array [12, 2] or None
            embed: Embedding vector or None
            seg_idx: Segmentation index
            seg_mat: Segmentation matrix or None
            seg_img: Rendered segmentation image or None

        Returns:
            Mock Detection object
        """
        detection = Mock()
        detection.frame = frame
        detection.pose_idx = pose_idx
        detection.pose = pose
        detection.embed = embed
        detection.seg_idx = seg_idx
        detection._seg_mat = seg_mat
        detection.seg_img = seg_img

        return detection

    return _create_mock_detection


@pytest.fixture
def sample_pose_data():
    """Generate sample pose data for testing."""

    def _generate_pose(
        center: tuple = (50, 50),
        valid_keypoints: int = 12,
        noise_scale: float = 5.0,
        seed: int = 42,
    ):
        """Generate a single pose with specified properties.

        Args:
            center: Center coordinates (x, y)
            valid_keypoints: Number of valid keypoints (0-12)
            noise_scale: Scale of random noise around center
            seed: Random seed for reproducibility

        Returns:
            Pose array of shape [12, 2]
        """
        np.random.seed(seed)
        pose = np.zeros((12, 2), dtype=np.float64)

        # Generate valid keypoints around center
        for i in range(valid_keypoints):
            pose[i] = [
                center[0] + np.random.normal(0, noise_scale),
                center[1] + np.random.normal(0, noise_scale),
            ]

        return pose

    return _generate_pose


@pytest.fixture
def sample_embedding_data():
    """Generate sample embedding data for testing."""

    def _generate_embedding(
        dim: int = 128,
        value: float | None = None,
        seed: int = 42,
    ):
        """Generate a single embedding vector.

        Args:
            dim: Embedding dimension
            value: Fixed value for all elements (random if None)
            seed: Random seed for reproducibility

        Returns:
            Embedding array of shape [dim]
        """
        if value is not None:
            return np.full(dim, value, dtype=np.float64)

        np.random.seed(seed)
        return np.random.random(dim).astype(np.float64)

    return _generate_embedding


@pytest.fixture
def sample_segmentation_data():
    """Generate sample segmentation data for testing."""

    def _generate_seg_mat(
        shape: tuple = (100, 100, 2),
        fill_value: int = 50,
        pad_value: int = -1,
        seed: int = 42,
    ):
        """Generate a segmentation matrix.

        Args:
            shape: Shape of segmentation matrix
            fill_value: Value for non-padded elements
            pad_value: Value for padded elements
            seed: Random seed for reproducibility

        Returns:
            Segmentation matrix array
        """
        np.random.seed(seed)
        seg_mat = np.full(shape, pad_value, dtype=np.int32)

        # Fill some non-padded values
        valid_points = shape[0] // 2
        for i in range(valid_points):
            seg_mat[i] = [
                fill_value + np.random.randint(-10, 10),
                fill_value + np.random.randint(-10, 10),
            ]

        return seg_mat

    return _generate_seg_mat


@pytest.fixture
def sample_seg_image():
    """Generate sample segmentation image for testing."""

    def _generate_seg_image(
        shape: tuple = (100, 100),
        center: tuple = (50, 50),
        radius: int = 20,
        seed: int = 42,
    ):
        """Generate a boolean segmentation image.

        Args:
            shape: Image shape (height, width)
            center: Center of filled circle
            radius: Radius of filled circle
            seed: Random seed for reproducibility

        Returns:
            Boolean segmentation image
        """
        np.random.seed(seed)
        img = np.zeros(shape, dtype=bool)

        # Create a circular mask
        y, x = np.ogrid[: shape[0], : shape[1]]
        mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
        img[mask] = True

        return img

    return _generate_seg_image


@pytest.fixture
def detection_factory(
    mock_detection, sample_pose_data, sample_embedding_data, sample_segmentation_data
):
    """Factory to create realistic mock Detection objects."""

    def _create_detection(
        frame: int = 0,
        pose_idx: int = 0,
        has_pose: bool = True,
        has_embedding: bool = True,
        has_segmentation: bool = True,
        pose_center: tuple = (50, 50),
        embed_dim: int = 128,
        embed_value: float | None = None,
        seg_shape: tuple = (100, 100, 2),
        seed: int | None = None,
    ):
        """Create a realistic mock Detection object.

        Args:
            frame: Frame index
            pose_idx: Pose index
            has_pose: Whether detection has pose data
            has_embedding: Whether detection has embedding data
            has_segmentation: Whether detection has segmentation data
            pose_center: Center for pose generation
            embed_dim: Embedding dimension
            embed_value: Fixed embedding value (random if None)
            seg_shape: Segmentation matrix shape
            seed: Random seed (derived from pose_idx if None)

        Returns:
            Mock Detection object with realistic data
        """
        if seed is None:
            seed = pose_idx + frame * 100

        # Generate pose data
        pose = sample_pose_data(center=pose_center, seed=seed) if has_pose else None

        # Generate embedding data
        embed = (
            sample_embedding_data(dim=embed_dim, value=embed_value, seed=seed)
            if has_embedding
            else None
        )

        # Generate segmentation data
        seg_mat = (
            sample_segmentation_data(shape=seg_shape, seed=seed)
            if has_segmentation
            else None
        )

        return mock_detection(
            frame=frame,
            pose_idx=pose_idx,
            pose=pose,
            embed=embed,
            seg_idx=pose_idx,
            seg_mat=seg_mat,
        )

    return _create_detection


@pytest.fixture
def features_factory(detection_factory):
    """Factory to create VectorizedDetectionFeatures objects."""

    def _create_features(
        n_detections: int = 3,
        pose_configs: list | None = None,
        embed_configs: list | None = None,
        seg_configs: list | None = None,
        seed: int = 42,
    ):
        """Create VectorizedDetectionFeatures with specified configurations.

        Args:
            n_detections: Number of detections to create
            pose_configs: List of pose configurations (has_pose, center)
            embed_configs: List of embedding configurations (has_embedding, dim, value)
            seg_configs: List of segmentation configurations (has_segmentation, shape)
            seed: Random seed for reproducibility

        Returns:
            VectorizedDetectionFeatures object
        """
        from mouse_tracking.matching.vectorized_features import (
            VectorizedDetectionFeatures,
        )

        detections = []

        for i in range(n_detections):
            # Configure pose
            if pose_configs and i < len(pose_configs):
                pose_config = pose_configs[i]
                has_pose = pose_config.get("has_pose", True)
                pose_center = pose_config.get("center", (50 + i * 10, 50 + i * 10))
            else:
                has_pose = True
                pose_center = (50 + i * 10, 50 + i * 10)

            # Configure embedding
            if embed_configs and i < len(embed_configs):
                embed_config = embed_configs[i]
                has_embedding = embed_config.get("has_embedding", True)
                embed_dim = embed_config.get("dim", 128)
                embed_value = embed_config.get("value", None)
            else:
                has_embedding = True
                embed_dim = 128
                embed_value = None

            # Configure segmentation
            if seg_configs and i < len(seg_configs):
                seg_config = seg_configs[i]
                has_segmentation = seg_config.get("has_segmentation", True)
                seg_shape = seg_config.get("shape", (100, 100, 2))
            else:
                has_segmentation = True
                seg_shape = (100, 100, 2)

            detection = detection_factory(
                frame=i,
                pose_idx=i,
                has_pose=has_pose,
                has_embedding=has_embedding,
                has_segmentation=has_segmentation,
                pose_center=pose_center,
                embed_dim=embed_dim,
                embed_value=embed_value,
                seg_shape=seg_shape,
                seed=seed + i,
            )

            detections.append(detection)

        return VectorizedDetectionFeatures(detections)

    return _create_features


@pytest.fixture
def array_equality_check():
    """Utility for checking array equality with NaN handling."""

    def _check_arrays_equal(arr1, arr2, rtol=1e-7, atol=1e-7):
        """Check if two arrays are equal, handling NaN values.

        Args:
            arr1: First array
            arr2: Second array
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            True if arrays are equal (considering NaN)
        """
        if arr1.shape != arr2.shape:
            return False

        # Check for NaN positions
        nan_mask1 = np.isnan(arr1)
        nan_mask2 = np.isnan(arr2)

        if not np.array_equal(nan_mask1, nan_mask2):
            return False

        # Check non-NaN values
        valid_mask = ~nan_mask1
        if np.any(valid_mask):
            return np.allclose(arr1[valid_mask], arr2[valid_mask], rtol=rtol, atol=atol)

        return True

    return _check_arrays_equal


@pytest.fixture
def performance_timer():
    """Utility for timing test operations."""
    import time

    def _time_operation(operation, *args, **kwargs):
        """Time a function call.

        Args:
            operation: Function to time
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            Tuple of (result, elapsed_time)
        """
        start_time = time.time()
        result = operation(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time

    return _time_operation
