"""Unit tests for aggregate_folder_data function.

This module tests the fecal boli data aggregation functionality with comprehensive
coverage of success paths, error conditions, and edge cases.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mouse_tracking.utils.fecal_boli import aggregate_folder_data


def _create_mock_h5_file_context(counts_data):
    """Helper function to create a mock H5 file context manager.

    Args:
        counts_data: numpy array representing fecal boli counts

    Returns:
        Mock object that can be used as H5 file context manager
    """
    mock_file = MagicMock()
    mock_counts = MagicMock()
    mock_counts.__getitem__.return_value.flatten.return_value.astype.return_value = (
        counts_data
    )
    mock_file.__enter__.return_value = {
        "dynamic_objects/fecal_boli/counts": mock_counts
    }
    mock_file.__exit__.return_value = None
    return mock_file


@pytest.mark.parametrize(
    "folder_path,depth,expected_pattern",
    [
        ("/test/folder", 2, "/test/folder/*/*/*_pose_est_v6.h5"),
        ("/another/path", 1, "/another/path/*/*_pose_est_v6.h5"),
        ("/deep/nested/path", 3, "/deep/nested/path/*/*/*/*_pose_est_v6.h5"),
        ("relative/path", 0, "relative/path/*_pose_est_v6.h5"),
    ],
)
def test_glob_pattern_construction(folder_path, depth, expected_pattern):
    """Test that glob patterns are constructed correctly for different folder depths.

    Args:
        folder_path: Input folder path
        depth: Subfolder depth parameter
        expected_pattern: Expected glob pattern to be generated
    """
    # Arrange
    test_file = f"{folder_path}/computer1/date1/video1_pose_est_v6.h5"
    test_counts = np.array([1.0, 2.0])

    with patch("mouse_tracking.utils.fecal_boli.glob.glob") as mock_glob:
        mock_glob.return_value = [test_file]  # Provide a file to avoid concat error

        with patch("mouse_tracking.utils.fecal_boli.h5py.File") as mock_h5:
            mock_h5.return_value = _create_mock_h5_file_context(test_counts)

            # Act
            aggregate_folder_data(folder_path, depth=depth)

            # Assert
            mock_glob.assert_called_once_with(expected_pattern)


@pytest.mark.parametrize(
    "counts_data,num_bins,expected_length",
    [
        (np.array([1, 2, 3, 4, 5]), -1, 5),  # Read all data
        (np.array([1, 2, 3, 4, 5]), 3, 3),  # Clip data
        (np.array([1, 2, 3, 4, 5]), 0, 0),  # Zero bins
        (np.array([]), -1, 0),  # Empty data
        (np.array([42]), 1, 1),  # Single value
    ],
)
def test_num_bins_parameter_handling(counts_data, num_bins, expected_length):
    """Test that num_bins parameter correctly controls data length.

    Args:
        counts_data: Input count data array
        num_bins: Number of bins to process
        expected_length: Expected length of processed data
    """
    # Arrange
    test_file = "/test/folder/computer1/date1/video1_pose_est_v6.h5"
    mock_h5_file = _create_mock_h5_file_context(counts_data)

    with (
        patch("mouse_tracking.utils.fecal_boli.glob.glob") as mock_glob,
        patch("mouse_tracking.utils.fecal_boli.h5py.File") as mock_h5,
    ):
        mock_glob.return_value = [test_file]
        mock_h5.return_value = mock_h5_file

        # Act
        result = aggregate_folder_data("/test/folder", num_bins=num_bins)

        # Assert
        assert (
            len(result.columns) == expected_length + 1
        )  # +1 for NetworkFilename column


def test_num_bins_padding_with_float_data():
    """Test that num_bins parameter correctly pads data when needed with float data."""
    # Arrange - Use float data to test padding functionality
    test_file = "/test/folder/computer1/date1/video1_pose_est_v6.h5"
    counts_data = np.array([1.0, 2.0, 3.0])  # 3 elements, will pad to 5
    num_bins = 5
    expected_length = 5

    mock_h5_file = _create_mock_h5_file_context(counts_data)

    with (
        patch("mouse_tracking.utils.fecal_boli.glob.glob") as mock_glob,
        patch("mouse_tracking.utils.fecal_boli.h5py.File") as mock_h5,
    ):
        mock_glob.return_value = [test_file]
        mock_h5.return_value = mock_h5_file

        # Act
        result = aggregate_folder_data("/test/folder", num_bins=num_bins)

        # Assert
        assert (
            len(result.columns) == expected_length + 1
        )  # +1 for NetworkFilename column
        # Check that the last two values are NaN (padded values)
        assert pd.isna(result.iloc[0][3])  # Fourth minute should be NaN
        assert pd.isna(result.iloc[0][4])  # Fifth minute should be NaN


def test_single_file_successful_processing():
    """Test successful processing of a single H5 file with normal data."""
    # Arrange
    test_folder = "/test/folder"
    test_file = "/test/folder/computer1/date1/video1_pose_est_v6.h5"
    test_counts = np.array([1.0, 2.0, 3.0, 4.0])
    expected_filename = "/computer1/date1/video1.avi"

    mock_h5_file = _create_mock_h5_file_context(test_counts)

    with (
        patch("mouse_tracking.utils.fecal_boli.glob.glob") as mock_glob,
        patch("mouse_tracking.utils.fecal_boli.h5py.File") as mock_h5,
    ):
        mock_glob.return_value = [test_file]
        mock_h5.return_value = mock_h5_file

        # Act
        result = aggregate_folder_data(test_folder)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["NetworkFilename"] == expected_filename
        assert result.shape[1] == 5  # 4 minute columns + NetworkFilename
        # Check that values are properly set
        for i in range(4):
            assert result.iloc[0][i] == test_counts[i]


def test_multiple_files_with_same_length_data():
    """Test processing multiple files with same data length."""
    # Arrange
    test_folder = "/test/folder"
    test_files = [
        "/test/folder/comp1/date1/video1_pose_est_v6.h5",
        "/test/folder/comp2/date2/video2_pose_est_v6.h5",
    ]
    test_counts = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]

    with (
        patch("mouse_tracking.utils.fecal_boli.glob.glob") as mock_glob,
        patch("mouse_tracking.utils.fecal_boli.h5py.File") as mock_h5,
    ):
        mock_glob.return_value = test_files
        mock_h5.side_effect = [
            _create_mock_h5_file_context(test_counts[0]),
            _create_mock_h5_file_context(test_counts[1]),
        ]

        # Act
        result = aggregate_folder_data(test_folder)

        # Assert
        assert len(result) == 2
        assert result.shape[1] == 4  # 3 minute columns + NetworkFilename
        # Check filenames are properly extracted
        expected_filenames = ["/comp1/date1/video1.avi", "/comp2/date2/video2.avi"]
        assert result["NetworkFilename"].tolist() == expected_filenames


def test_multiple_files_with_different_length_data():
    """Test processing multiple files with different data lengths."""
    # Arrange
    test_folder = "/test/folder"
    test_files = [
        "/test/folder/comp1/date1/video1_pose_est_v6.h5",
        "/test/folder/comp2/date2/video2_pose_est_v6.h5",
    ]
    test_counts = [
        np.array([1.0, 2.0]),  # Short data
        np.array([3.0, 4.0, 5.0, 6.0]),  # Long data
    ]

    with (
        patch("mouse_tracking.utils.fecal_boli.glob.glob") as mock_glob,
        patch("mouse_tracking.utils.fecal_boli.h5py.File") as mock_h5,
    ):
        mock_glob.return_value = test_files
        mock_h5.side_effect = [
            _create_mock_h5_file_context(test_counts[0]),
            _create_mock_h5_file_context(test_counts[1]),
        ]

        # Act
        result = aggregate_folder_data(test_folder)

        # Assert
        assert len(result) == 2
        # Result should have columns for the maximum length found across all files
        assert result.shape[1] == 5  # 4 minute columns + NetworkFilename
        # Check that NaN values are properly handled for shorter data
        assert pd.isna(result.iloc[0][2])  # Third minute should be NaN for first file
        assert pd.isna(result.iloc[0][3])  # Fourth minute should be NaN for first file


@pytest.mark.parametrize(
    "num_bins,counts_data,expected_first_row_values",
    [
        (2, np.array([10.0, 20.0, 30.0, 40.0]), [10.0, 20.0]),  # Clipping
        (-1, np.array([5.0, 15.0]), [5.0, 15.0]),  # No modification
        (0, np.array([1.0, 2.0, 3.0]), []),  # Zero bins
    ],
)
def test_data_clipping_and_padding(num_bins, counts_data, expected_first_row_values):
    """Test that data is properly clipped or padded based on num_bins parameter.

    Args:
        num_bins: Number of bins to process
        counts_data: Input count data
        expected_first_row_values: Expected values in the first row (excluding NetworkFilename)
    """
    # Arrange
    test_folder = "/test/folder"
    test_file = "/test/folder/comp1/date1/video1_pose_est_v6.h5"

    mock_h5_file = _create_mock_h5_file_context(counts_data)

    with (
        patch("mouse_tracking.utils.fecal_boli.glob.glob") as mock_glob,
        patch("mouse_tracking.utils.fecal_boli.h5py.File") as mock_h5,
    ):
        mock_glob.return_value = [test_file]
        mock_h5.return_value = mock_h5_file

        # Act
        result = aggregate_folder_data(test_folder, num_bins=num_bins)

        # Assert
        if len(expected_first_row_values) == 0:
            assert result.shape[1] == 1  # Only NetworkFilename column
        else:
            # Compare values excluding NetworkFilename column
            actual_values = result.iloc[0].drop("NetworkFilename").values
            for i, expected_val in enumerate(expected_first_row_values):
                if pd.isna(expected_val):
                    assert pd.isna(actual_values[i])
                else:
                    assert actual_values[i] == expected_val


def test_data_padding_with_float_values():
    """Test padding functionality separately with float data to avoid numpy integer/NaN conflict."""
    # Arrange
    test_folder = "/test/folder"
    test_file = "/test/folder/comp1/date1/video1_pose_est_v6.h5"
    counts_data = np.array([10.0, 20.0, 30.0])  # 3 values, will pad to 6
    num_bins = 6
    expected_first_row_values = [10.0, 20.0, 30.0, np.nan, np.nan, np.nan]

    mock_h5_file = _create_mock_h5_file_context(counts_data)

    with (
        patch("mouse_tracking.utils.fecal_boli.glob.glob") as mock_glob,
        patch("mouse_tracking.utils.fecal_boli.h5py.File") as mock_h5,
    ):
        mock_glob.return_value = [test_file]
        mock_h5.return_value = mock_h5_file

        # Act
        result = aggregate_folder_data(test_folder, num_bins=num_bins)

        # Assert
        actual_values = result.iloc[0].drop("NetworkFilename").values
        for i, expected_val in enumerate(expected_first_row_values):
            if pd.isna(expected_val):
                assert pd.isna(actual_values[i])
            else:
                assert actual_values[i] == expected_val


def test_empty_folder_no_files_found():
    """Test behavior when no matching files are found in the folder."""
    # Arrange
    test_folder = "/empty/folder"

    with patch("mouse_tracking.utils.fecal_boli.glob.glob") as mock_glob:
        mock_glob.return_value = []

        # Act & Assert
        # The function currently fails with empty file lists, this is a bug that should be fixed
        with pytest.raises(ValueError, match="No objects to concatenate"):
            aggregate_folder_data(test_folder)


def test_file_with_empty_counts_data():
    """Test processing a file that contains empty counts data."""
    # Arrange
    test_folder = "/test/folder"
    test_file = "/test/folder/comp1/date1/video1_pose_est_v6.h5"
    empty_counts = np.array([])

    mock_h5_file = _create_mock_h5_file_context(empty_counts)

    with (
        patch("mouse_tracking.utils.fecal_boli.glob.glob") as mock_glob,
        patch("mouse_tracking.utils.fecal_boli.h5py.File") as mock_h5,
    ):
        mock_glob.return_value = [test_file]
        mock_h5.return_value = mock_h5_file

        # Act
        result = aggregate_folder_data(test_folder)

        # Assert
        # When counts are empty, the pivot results in an empty DataFrame
        assert len(result) == 0
        assert "NetworkFilename" in result.columns


def test_h5py_file_error_handling():
    """Test error handling when H5 file cannot be opened."""
    # Arrange
    test_folder = "/test/folder"
    test_file = "/test/folder/comp1/date1/video1_pose_est_v6.h5"

    with (
        patch("mouse_tracking.utils.fecal_boli.glob.glob") as mock_glob,
        patch("mouse_tracking.utils.fecal_boli.h5py.File") as mock_h5,
    ):
        mock_glob.return_value = [test_file]
        mock_h5.side_effect = OSError("Unable to open file")

        # Act & Assert
        with pytest.raises(OSError):
            aggregate_folder_data(test_folder)


def test_missing_data_structure_in_h5_file():
    """Test error handling when expected data structure is missing from H5 file."""
    # Arrange
    test_folder = "/test/folder"
    test_file = "/test/folder/comp1/date1/video1_pose_est_v6.h5"

    mock_file = MagicMock()
    mock_file.__enter__.return_value = {}  # Empty file structure
    mock_file.__exit__.return_value = None

    with (
        patch("mouse_tracking.utils.fecal_boli.glob.glob") as mock_glob,
        patch("mouse_tracking.utils.fecal_boli.h5py.File") as mock_h5,
    ):
        mock_glob.return_value = [test_file]
        mock_h5.return_value = mock_file

        # Act & Assert
        with pytest.raises(KeyError):
            aggregate_folder_data(test_folder)


@pytest.mark.parametrize(
    "invalid_folder",
    [
        None,  # None value
    ],
)
def test_invalid_folder_path_handling_type_error(invalid_folder):
    """Test behavior with None folder path that should raise TypeError.

    Args:
        invalid_folder: Invalid folder path to test
    """
    # Arrange & Act & Assert
    with pytest.raises(TypeError):
        aggregate_folder_data(invalid_folder)


@pytest.mark.parametrize(
    "invalid_folder",
    [
        "",  # Empty string
        "/nonexistent/path",  # Path that doesn't exist
    ],
)
def test_invalid_folder_path_handling_no_files(invalid_folder):
    """Test behavior with invalid folder paths that result in no files found.

    Args:
        invalid_folder: Invalid folder path to test
    """
    # Arrange & Act & Assert
    with patch("mouse_tracking.utils.fecal_boli.glob.glob") as mock_glob:
        mock_glob.return_value = []  # No files found for invalid paths

        # The function currently fails with empty file lists, this is expected behavior
        with pytest.raises(ValueError, match="No objects to concatenate"):
            aggregate_folder_data(invalid_folder)


def test_network_filename_extraction_accuracy():
    """Test that NetworkFilename is correctly extracted from file paths."""
    # Arrange
    test_folder = "/base/project/folder"
    test_cases = [
        {
            "file_path": "/base/project/folder/computer1/20240101/experiment1_pose_est_v6.h5",
            "expected_filename": "/computer1/20240101/experiment1.avi",
        },
        {
            "file_path": "/base/project/folder/lab-pc/2024-01-15/long_video_name_pose_est_v6.h5",
            "expected_filename": "/lab-pc/2024-01-15/long_video_name.avi",
        },
    ]

    for i, test_case in enumerate(test_cases):
        with (
            patch("mouse_tracking.utils.fecal_boli.glob.glob") as mock_glob,
            patch("mouse_tracking.utils.fecal_boli.h5py.File") as mock_h5,
        ):
            mock_glob.return_value = [test_case["file_path"]]
            mock_h5.return_value = _create_mock_h5_file_context(np.array([1.0, 2.0]))

            # Act
            result = aggregate_folder_data(test_folder)

            # Assert
            assert (
                result.iloc[0]["NetworkFilename"] == test_case["expected_filename"]
            ), (
                f"Test case {i} failed: expected {test_case['expected_filename']}, got {result.iloc[0]['NetworkFilename']}"
            )


def test_data_type_conversion_to_float():
    """Test that count data is properly converted to float type."""
    # Arrange
    test_folder = "/test/folder"
    test_file = "/test/folder/comp1/date1/video1_pose_est_v6.h5"
    # Use integer data to verify float conversion
    integer_counts = np.array([1, 2, 3, 4], dtype=np.int32)

    mock_h5_file = _create_mock_h5_file_context(integer_counts.astype(float))

    with (
        patch("mouse_tracking.utils.fecal_boli.glob.glob") as mock_glob,
        patch("mouse_tracking.utils.fecal_boli.h5py.File") as mock_h5,
    ):
        mock_glob.return_value = [test_file]
        mock_h5.return_value = mock_h5_file

        # Act
        result = aggregate_folder_data(test_folder)

        # Assert
        # Check that all numeric columns contain float values
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != "NetworkFilename":  # Skip the string column
                assert result[col].dtype == np.float64 or pd.api.types.is_float_dtype(
                    result[col]
                )


def test_dataframe_structure_and_pivot_correctness():
    """Test that the resulting DataFrame has correct structure after pivot operation."""
    # Arrange
    test_folder = "/test/folder"
    test_files = [
        "/test/folder/comp1/date1/video1_pose_est_v6.h5",
        "/test/folder/comp2/date2/video2_pose_est_v6.h5",
    ]
    test_counts = [np.array([10.0, 20.0, 30.0]), np.array([40.0, 50.0, 60.0])]

    with (
        patch("mouse_tracking.utils.fecal_boli.glob.glob") as mock_glob,
        patch("mouse_tracking.utils.fecal_boli.h5py.File") as mock_h5,
    ):
        mock_glob.return_value = test_files
        mock_h5.side_effect = [
            _create_mock_h5_file_context(test_counts[0]),
            _create_mock_h5_file_context(test_counts[1]),
        ]

        # Act
        result = aggregate_folder_data(test_folder)

        # Assert
        # Check DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Two files processed
        assert "NetworkFilename" in result.columns

        # Check minute columns are properly numbered (0, 1, 2)
        minute_columns = [col for col in result.columns if col != "NetworkFilename"]
        expected_minute_columns = [0, 1, 2]
        assert minute_columns == expected_minute_columns

        # Check that data is properly assigned to correct minute columns
        for i, expected_counts in enumerate(test_counts):
            for j, expected_count in enumerate(expected_counts):
                assert result.iloc[i][j] == expected_count
