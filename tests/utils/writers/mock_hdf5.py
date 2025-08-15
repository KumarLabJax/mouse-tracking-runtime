"""Test helpers related to HDF5 files."""


class MockAttrs:
    """Mock class that supports item assignment for HDF5 attrs."""

    def __init__(self, initial_data=None):
        self._data = initial_data or {}

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        """Get a value from the attrs dictionary with optional default."""
        return self._data.get(key, default)


def create_mock_h5_context(
    existing_datasets=None, pose_data_shape=None, seg_data_shape=None
):
    """Helper function to create a mock H5 file context manager.

    Args:
        existing_datasets: List of dataset names that already exist in the file
        pose_data_shape: Shape of the pose data for validation
        seg_data_shape: Shape of the segmentation data for validation

    Returns:
        Mock object that can be used as H5 file context manager
    """
    from unittest.mock import Mock

    mock_context = Mock()
    mock_context.__enter__ = Mock(return_value=mock_context)
    mock_context.__exit__ = Mock(return_value=None)

    # Track which datasets exist and their deletion (for compatibility with existing tests)
    mock_context._datasets = dict.fromkeys(existing_datasets or [], Mock())
    mock_context._deleted_datasets = []

    # Track created datasets (enhanced functionality)
    created_datasets = {}
    deleted_datasets = []

    def mock_create_dataset(path, data=None, **kwargs):
        mock_dataset = Mock()
        mock_dataset.attrs = MockAttrs()
        created_datasets[path] = {
            "dataset": mock_dataset,
            "data": data,
            "kwargs": kwargs,
        }
        # Also track in _datasets for compatibility
        mock_context._datasets[path] = mock_dataset
        if path in mock_context._deleted_datasets:
            mock_context._deleted_datasets.remove(path)
        return mock_dataset

    def mock_getitem(key):
        if key == "poseest/points" and pose_data_shape is not None:
            mock_pose_dataset = Mock()
            mock_pose_dataset.shape = pose_data_shape
            return mock_pose_dataset
        if key == "poseest/seg_data" and seg_data_shape is not None:
            mock_seg_dataset = Mock()
            mock_seg_dataset.shape = seg_data_shape
            return mock_seg_dataset
        if key in created_datasets:
            return created_datasets[key]["dataset"]
        if key in mock_context._datasets:
            return mock_context._datasets[key]
        raise KeyError(f"Dataset {key} not found")

    def mock_contains(key):
        # Check if key exists in either the initial existing_datasets or in _datasets
        in_existing = key in (existing_datasets or [])
        in_datasets = key in mock_context._datasets
        not_deleted = key not in mock_context._deleted_datasets
        return (in_existing or in_datasets) and not_deleted

    def mock_delitem(key):
        deleted_datasets.append(key)
        mock_context._deleted_datasets.append(key)

    # Use Mock objects instead of functions to preserve call tracking
    mock_context.create_dataset = Mock(side_effect=mock_create_dataset)
    mock_context.__getitem__ = Mock(side_effect=mock_getitem)
    mock_context.__contains__ = Mock(side_effect=mock_contains)
    mock_context.__delitem__ = Mock(side_effect=mock_delitem)

    # Expose tracking data
    mock_context.created_datasets = created_datasets
    mock_context.deleted_datasets = deleted_datasets

    return mock_context
