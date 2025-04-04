#!/usr/bin/env python3
"""Process JABS postprocessing summary tables by aggregating behavior data into bins.

This module provides functionality to read, process, and aggregate behavior data
from JABS postprocessing summary tables, calculating metrics like time spent in
behaviors and distances traveled.
"""
import argparse

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process JABS postprocessing summary table"
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="input JABS-postprocessing summary table file",
    )
    parser.add_argument(
        "-b", "--bin_size", type=int, required=True, help="number of bins to aggregate"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="output file name"
    )
    return parser.parse_args()


def extract_behavior_name(file_path: str) -> str:
    """Extract behavior name from the first row of the file.

    Args:
        file_path: Path to the input file.

    Returns:
        str: Behavior name.
    """
    data_header = pd.read_csv(file_path, nrows=1)
    return data_header["Behavior"].iloc[0]


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from the file, skipping header rows.

    Args:
        file_path: Path to the input file.

    Returns:
        pd.DataFrame: Dataframe containing the loaded data.
    """
    return pd.read_csv(file_path, skiprows=2)


def preprocess_data(data: pd.DataFrame, behavior: str) -> pd.DataFrame:
    """Preprocess the data by renaming and modifying columns.

    Args:
        data: Input dataframe.
        behavior: Behavior name to use as prefix.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    # Drop longterm_idx column if it exists
    data = data.drop(columns=["longterm_idx"], errors="ignore")

    # Rename exp_prefix to MouseID
    data = data.rename(columns={"exp_prefix": "MouseID"})

    # Rename all columns except MouseID by prefixing with behavior
    renamed_columns = {
        col: f"{behavior}_{col}" for col in data.columns if col != "MouseID"
    }

    return data.rename(columns=renamed_columns)


def get_columns_to_exclude(behavior: str) -> list:
    """Get list of columns to exclude from final output.

    Args:
        behavior: Behavior name.

    Returns:
        list: List of column names to exclude.
    """
    suffixes = [
        "time_no_pred",
        "time_not_behavior",
        "time_behavior",
        "bout_behavior",
        "not_behavior_dist",
        "behavior_dist",
    ]
    return [f"{behavior}_{suffix}" for suffix in suffixes]


def aggregate_data_by_bin_size(
    data: pd.DataFrame, bin_size: int, behavior: str
) -> pd.DataFrame:
    """Aggregate data by bin size.

    Args:
        data: Preprocessed dataframe.
        bin_size: Number of bins to aggregate.
        behavior: Behavior name.

    Returns:
        pd.DataFrame: Aggregated dataframe.
    """
    # Group by MouseID and take only the first bin_size rows for each mouse
    grouped = data.groupby("MouseID")
    filtered_data = pd.concat([group.iloc[:bin_size] for _, group in grouped])

    # Aggregate numeric columns by summing them
    numeric_cols = filtered_data.select_dtypes(include=["number"]).columns
    aggregated = filtered_data.groupby("MouseID")[numeric_cols].sum()

    # Add calculated columns
    time_behavior_col = f"{behavior}_time_behavior"
    time_not_behavior_col = f"{behavior}_time_not_behavior"
    behavior_dist_col = f"{behavior}_behavior_dist"

    # Calculate time spent in behavior
    # TODO: Do we need to make `5` a configurable parameter?
    aggregated[f"bin_avg_{bin_size*5}.{behavior}_time_secs"] = (
        aggregated[time_behavior_col]
        / (aggregated[time_behavior_col] + aggregated[time_not_behavior_col])
        * bin_size
        * 5
    )

    # Calculate average distance (in cm)
    # TODO: Do we need to make `5` a configurable parameter?
    aggregated[f"bin_avg_{bin_size*5}.{behavior}_distance_cm"] = aggregated[
        behavior_dist_col
    ] / (bin_size * 5)

    # Reset index to make MouseID a regular column
    return aggregated.reset_index()


def main():
    """Main function to process JABS postprocessing summary table."""
    args = parse_args()

    # Extract behavior name and load data
    behavior = extract_behavior_name(args.file)
    data = load_data(args.file)

    # Preprocess data
    processed_data = preprocess_data(data, behavior)

    # Get columns to exclude
    cols_to_exclude = get_columns_to_exclude(behavior)

    # Aggregate data by bin size
    aggregated_data = aggregate_data_by_bin_size(
        processed_data, args.bin_size, behavior
    )

    # Drop excluded columns
    final_data = aggregated_data.drop(columns=cols_to_exclude, errors="ignore")

    # Write to CSV
    final_data.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
