"""Unit tests for aggregate_fecal_boli CLI command."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from mouse_tracking.cli.utils import aggregate_fecal_boli, app


@pytest.fixture
def runner():
    """Provide a CliRunner instance for testing."""
    return CliRunner()


@pytest.fixture
def sample_dataframe():
    """Provide a sample DataFrame to mock the fecal_boli.aggregate_folder_data return value."""
    mock_df = MagicMock(spec=pd.DataFrame)
    mock_df.to_csv = MagicMock()
    return mock_df


@pytest.fixture
def temp_folder():
    """Provide a temporary folder for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_output_file():
    """Provide a temporary output file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        yield Path(temp_file.name)
    # Cleanup handled by tempfile


class TestAggregateFecalBoli:
    """Test class for aggregate_fecal_boli CLI command."""

    def test_function_exists_and_is_callable(self):
        """Test that aggregate_fecal_boli function exists and is callable."""
        # Arrange & Act & Assert
        assert callable(aggregate_fecal_boli)

    @patch("mouse_tracking.cli.utils.fecal_boli.aggregate_folder_data")
    def test_successful_execution_with_defaults(
        self, mock_aggregate, sample_dataframe, temp_folder, temp_output_file, runner
    ):
        """Test successful execution with default parameters."""
        # Arrange
        mock_aggregate.return_value = sample_dataframe

        # Act
        result = runner.invoke(
            app,
            [
                "aggregate-fecal-boli",
                str(temp_folder),
                "--output",
                str(temp_output_file),
            ],
        )

        # Assert
        assert result.exit_code == 0
        mock_aggregate.assert_called_once_with(str(temp_folder), depth=2, num_bins=-1)
        sample_dataframe.to_csv.assert_called_once_with(temp_output_file, index=False)

    @patch("mouse_tracking.cli.utils.fecal_boli.aggregate_folder_data")
    def test_execution_with_custom_parameters(
        self, mock_aggregate, sample_dataframe, temp_folder, temp_output_file, runner
    ):
        """Test execution with custom parameters."""
        # Arrange
        mock_aggregate.return_value = sample_dataframe
        custom_depth = 3
        custom_num_bins = 5

        # Act
        result = runner.invoke(
            app,
            [
                "aggregate-fecal-boli",
                str(temp_folder),
                "--folder-depth",
                str(custom_depth),
                "--num-bins",
                str(custom_num_bins),
                "--output",
                str(temp_output_file),
            ],
        )

        # Assert
        assert result.exit_code == 0
        mock_aggregate.assert_called_once_with(
            str(temp_folder), depth=custom_depth, num_bins=custom_num_bins
        )

    @patch("mouse_tracking.cli.utils.fecal_boli.aggregate_folder_data")
    def test_default_output_filename(
        self, mock_aggregate, sample_dataframe, temp_folder, runner
    ):
        """Test that default output filename is used when not specified."""
        # Arrange
        mock_aggregate.return_value = sample_dataframe

        with patch("pathlib.Path.exists", return_value=False):  # Avoid file conflicts
            # Act
            result = runner.invoke(app, ["aggregate-fecal-boli", str(temp_folder)])

        # Assert
        assert result.exit_code == 0
        sample_dataframe.to_csv.assert_called_once_with(Path("output.csv"), index=False)

    @patch("mouse_tracking.cli.utils.fecal_boli.aggregate_folder_data")
    def test_parameter_type_conversion(
        self, mock_aggregate, sample_dataframe, temp_folder, temp_output_file, runner
    ):
        """Test that parameters are properly converted to correct types."""
        # Arrange
        mock_aggregate.return_value = sample_dataframe

        # Act
        result = runner.invoke(
            app,
            [
                "aggregate-fecal-boli",
                str(temp_folder),
                "--folder-depth",
                "1",
                "--num-bins",
                "10",
                "--output",
                str(temp_output_file),
            ],
        )

        # Assert
        assert result.exit_code == 0
        mock_aggregate.assert_called_once_with(
            str(temp_folder),
            depth=1,  # Should be int
            num_bins=10,  # Should be int
        )

    @patch("mouse_tracking.cli.utils.fecal_boli.aggregate_folder_data")
    def test_folder_path_conversion_to_string(
        self, mock_aggregate, sample_dataframe, temp_folder, temp_output_file, runner
    ):
        """Test that folder Path is properly converted to string."""
        # Arrange
        mock_aggregate.return_value = sample_dataframe

        # Act
        result = runner.invoke(
            app,
            [
                "aggregate-fecal-boli",
                str(temp_folder),
                "--output",
                str(temp_output_file),
            ],
        )

        # Assert
        assert result.exit_code == 0
        # Verify that the folder argument was converted to string
        args, kwargs = mock_aggregate.call_args
        assert isinstance(args[0], str)
        assert args[0] == str(temp_folder)

    @patch("mouse_tracking.cli.utils.fecal_boli.aggregate_folder_data")
    def test_aggregate_folder_data_exception_handling(
        self, mock_aggregate, temp_folder, temp_output_file, runner
    ):
        """Test handling of exceptions from aggregate_folder_data."""
        # Arrange
        mock_aggregate.side_effect = ValueError("No objects to concatenate")

        # Act
        result = runner.invoke(
            app,
            [
                "aggregate-fecal-boli",
                str(temp_folder),
                "--output",
                str(temp_output_file),
            ],
        )

        # Assert
        assert result.exit_code != 0
        # Exception should be raised and caught by typer, resulting in non-zero exit
        assert isinstance(result.exception, ValueError)
        assert str(result.exception) == "No objects to concatenate"

    @patch("mouse_tracking.cli.utils.fecal_boli.aggregate_folder_data")
    def test_csv_write_exception_handling(self, mock_aggregate, temp_folder, runner):
        """Test handling of exceptions during CSV writing."""
        # Arrange
        failing_df = MagicMock(spec=pd.DataFrame)
        failing_df.to_csv.side_effect = PermissionError("Permission denied")
        mock_aggregate.return_value = failing_df

        # Act
        result = runner.invoke(
            app,
            [
                "aggregate-fecal-boli",
                str(temp_folder),
                "--output",
                "/invalid/path/output.csv",
            ],
        )

        # Assert
        assert result.exit_code != 0

    def test_missing_required_folder_argument(self, runner):
        """Test behavior when required folder argument is missing."""
        # Arrange & Act
        result = runner.invoke(app, ["aggregate-fecal-boli"])

        # Assert
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout

    @pytest.mark.parametrize(
        "folder_depth,num_bins,expected_depth,expected_bins",
        [
            ("0", "-1", 0, -1),
            ("1", "0", 1, 0),
            ("5", "100", 5, 100),
            ("-1", "-1", -1, -1),  # Edge case: negative depth
        ],
        ids=[
            "zero_depth_all_bins",
            "one_depth_zero_bins",
            "large_values",
            "negative_depth",
        ],
    )
    @patch("mouse_tracking.cli.utils.fecal_boli.aggregate_folder_data")
    def test_parameter_edge_cases(
        self,
        mock_aggregate,
        sample_dataframe,
        folder_depth,
        num_bins,
        expected_depth,
        expected_bins,
        temp_folder,
        temp_output_file,
        runner,
    ):
        """Test edge cases for folder_depth and num_bins parameters."""
        # Arrange
        mock_aggregate.return_value = sample_dataframe

        # Act
        result = runner.invoke(
            app,
            [
                "aggregate-fecal-boli",
                str(temp_folder),
                "--folder-depth",
                folder_depth,
                "--num-bins",
                num_bins,
                "--output",
                str(temp_output_file),
            ],
        )

        # Assert
        assert result.exit_code == 0
        mock_aggregate.assert_called_once_with(
            str(temp_folder), depth=expected_depth, num_bins=expected_bins
        )

    def test_help_message_content(self, runner):
        """Test that help message contains expected content."""
        # Arrange & Act
        result = runner.invoke(
            app, ["aggregate-fecal-boli", "--help"], env={"TERM": "dumb"}
        )

        # Assert
        assert result.exit_code == 0
        assert "Aggregate fecal boli data" in result.stdout
        assert "--folder-depth" in result.stdout
        assert "--num-bins" in result.stdout
        assert "--output" in result.stdout
        assert "Path to the folder containing fecal boli data" in result.stdout

    @patch("mouse_tracking.cli.utils.fecal_boli.aggregate_folder_data")
    def test_relative_path_handling(self, mock_aggregate, sample_dataframe, runner):
        """Test handling of relative paths."""
        # Arrange
        mock_aggregate.return_value = sample_dataframe
        relative_folder = "data/fecal_boli"

        with patch("pathlib.Path.exists", return_value=False):
            # Act
            result = runner.invoke(app, ["aggregate-fecal-boli", relative_folder])

        # Assert
        assert result.exit_code == 0
        mock_aggregate.assert_called_once_with(relative_folder, depth=2, num_bins=-1)

    @patch("mouse_tracking.cli.utils.fecal_boli.aggregate_folder_data")
    def test_output_file_with_different_extensions(
        self, mock_aggregate, sample_dataframe, temp_folder, runner
    ):
        """Test that output works with different file extensions."""
        # Arrange
        mock_aggregate.return_value = sample_dataframe

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            output_file = Path(temp_file.name)

        # Act
        result = runner.invoke(
            app,
            ["aggregate-fecal-boli", str(temp_folder), "--output", str(output_file)],
        )

        # Assert
        assert result.exit_code == 0
        sample_dataframe.to_csv.assert_called_once_with(output_file, index=False)

    @patch("mouse_tracking.cli.utils.fecal_boli.aggregate_folder_data")
    def test_dataframe_to_csv_parameters(
        self, mock_aggregate, sample_dataframe, temp_folder, temp_output_file, runner
    ):
        """Test that DataFrame.to_csv is called with correct parameters."""
        # Arrange
        mock_aggregate.return_value = sample_dataframe

        # Act
        result = runner.invoke(
            app,
            [
                "aggregate-fecal-boli",
                str(temp_folder),
                "--output",
                str(temp_output_file),
            ],
        )

        # Assert
        assert result.exit_code == 0
        # Verify to_csv is called with index=False
        sample_dataframe.to_csv.assert_called_once_with(temp_output_file, index=False)

    @pytest.mark.parametrize(
        "invalid_num_bins",
        [
            "invalid",
            "1.5",
            "abc",
        ],
        ids=["non_numeric_string", "float_string", "alphabetic_string"],
    )
    def test_invalid_num_bins_parameter(self, invalid_num_bins, temp_folder, runner):
        """Test behavior with invalid num_bins parameter values."""
        # Arrange & Act
        result = runner.invoke(
            app,
            ["aggregate-fecal-boli", str(temp_folder), "--num-bins", invalid_num_bins],
        )

        # Assert
        assert result.exit_code != 0
        assert "Invalid value" in result.stdout or "invalid literal" in result.stdout

    @pytest.mark.parametrize(
        "invalid_folder_depth",
        [
            "invalid",
            "2.7",
            "xyz",
        ],
        ids=["non_numeric_string", "float_string", "alphabetic_string"],
    )
    def test_invalid_folder_depth_parameter(
        self, invalid_folder_depth, temp_folder, runner
    ):
        """Test behavior with invalid folder_depth parameter values."""
        # Arrange & Act
        result = runner.invoke(
            app,
            [
                "aggregate-fecal-boli",
                str(temp_folder),
                "--folder-depth",
                invalid_folder_depth,
            ],
        )

        # Assert
        assert result.exit_code != 0
        assert "Invalid value" in result.stdout or "invalid literal" in result.stdout

    @patch("mouse_tracking.cli.utils.fecal_boli.aggregate_folder_data")
    def test_empty_dataframe_handling(
        self, mock_aggregate, temp_folder, temp_output_file, runner
    ):
        """Test handling of empty DataFrame returned by aggregate_folder_data."""
        # Arrange
        empty_df = MagicMock(spec=pd.DataFrame)
        empty_df.to_csv = MagicMock()
        mock_aggregate.return_value = empty_df

        # Act
        result = runner.invoke(
            app,
            [
                "aggregate-fecal-boli",
                str(temp_folder),
                "--output",
                str(temp_output_file),
            ],
        )

        # Assert
        assert result.exit_code == 0
        empty_df.to_csv.assert_called_once_with(temp_output_file, index=False)
