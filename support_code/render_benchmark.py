#!/usr/bin/env python3
"""Render performance benchmark results.

This script processes benchmark results from a delimited trace file and generates visualizations
to analyze the performance of various tasks. It reads data from `trace_tabbed.csv`,
performs data cleaning and transformation, and creates plots using the `plotnine` library.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9
from sklearn.linear_model import LinearRegression


def parse_trace(file: Path, delimiter: str = "\t") -> pd.DataFrame:
    r"""Parse the trace file and return a DataFrame with relevant information.

    Args:
        file: Input trace file path.
        delimiter: Delimiter used in the input file.

    Returns:
        Dataframe containing the parsed trace data.

    Notes:
        The required columns are: task_id, process, duration, start, complete, pcpu, and peak_rss.
        Columns are covered from a standard `-with-trace` nextflow output.

        If script is included, you will need to remove newlines and tabs from this column.
        nextflow log [nextflow_job_id] -f task_id,process,duration,start,complete,pcpu,peak_rss,script > trace.tsv
        awk 'BEGIN {FS="\t";OFS="|"} { if (NF==8) { print LASTROW, SCRIPT; LASTROW=$1 OFS $2 OFS $3 OFS $4 OFS $5 OFS $6 OFS $7; SCRIPT=$8} else {ADD_SCRIPT=$0; SCRIPT=SCRIPT FS ADD_SCRIPT }} END { print LASTROW, SCRIPT }' trace.tsv  > trace_tabbed.csv

    Todo:
        Add proper error handling for missing columns.
    """
    # Read the CSV file into a DataFrame
    return_df = pd.read_csv(file, sep=delimiter)

    # Rename some columns that are different between querying the log vs exporting trace directly
    return_df = return_df.rename(columns={r"%cpu": "pcpu", "name": "process"})

    # Extract branch and task from the process column
    return_df["branch"] = [x.split(":")[0] for x in return_df["process"]]
    return_df["task"] = [x.split(":")[1].split(" ")[0] for x in return_df["process"]]

    # Convert pcpu to float
    return_df["pcpu"] = return_df["pcpu"].str.replace("%", "").astype(float)

    # Convert peak_rss shorthand to bytes
    unit_conversions = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "NaN": 0}
    return_df["rss_coeff"] = return_df["peak_rss"].str.extract(r"(\d+)").astype(float)
    return_df["rss_unit"] = return_df["peak_rss"].str.extract(r"([KMG]B)")
    return_df["peak_rss"] = return_df["rss_coeff"] * return_df["rss_unit"].map(
        unit_conversions
    )

    # Convert duration to seconds
    return_df["duration"] = pd.to_timedelta(return_df["duration"]).dt.total_seconds()

    return return_df


# Dataframe is read in from an export from nextflow trace
# Read in the trace from 5, 10, 20, and 30 minute scaling test
scaling_df = parse_trace("trace_tabbed.csv", delimiter="|")
# Extract video duration from the script column
tmp = [re.search(r"(5|10|20|30)_min", x) for x in scaling_df["script"]]
scaling_df["video_duration_min"] = [
    int(x.groups(0)[0]) if x is not None else 0 for x in tmp
]

# Read in the trace from a 1-hr test batch
hr_df = parse_trace("trace_1hr_batch.txt")
hr_df["video_duration_min"] = 60

cols_of_interest = ["duration", "pcpu", "peak_rss"]
df_melted = pd.melt(
    pd.concat([scaling_df, hr_df]),
    id_vars=["video_duration_min", "process", "branch", "task"],
    value_vars=cols_of_interest,
)

for grp, df_subset in df_melted.groupby(["task", "variable"]):
    # Ignore a handful of tasks that should be trivial
    if grp[0] in (
        "ADD_EXAMPLES_TO_SLEAP",
        "ADD_VERSION_FBOLI",
        "ADD_VERSION_GAIT",
        "ADD_VERSION_JABS",
        "ADD_VERSION_MORPH",
        "ADD_VERSION_QC",
        "CHECK_FILE",
        "DELETE_DEFAULT_FBOLI",
        "DELETE_DEFAULT_JABS",
        "GET_WORKFLOW_VERSION",
        "MERGE_DIST_AC",
        "MERGE_DIST_B",
        "MERGE_FEATURE_COLS",
        "MERGE_FECAL_BOLI",
        "MERGE_GAIT",
        "MERGE_JABS",
        "MERGE_REAR_PAW_WIDTHS",
        "NOURL_FBOLI",
        "NOURL_GAIT",
        "NOURL_JABS",
        "NOURL_MORPH",
        "NOURL_QC",
        "PUBLISH_FBOLI",
        "PUBLISH_GAIT",
        "PUBLISH_MORPHOMETRICS",
        "PUBLISH_SM_MANUAL_CORRECT",
        "PUBLISH_SM_POSE_V2",
        "PUBLISH_SM_POSE_V6",
        "PUBLISH_SM_POSE_V6_NOCORN",
        "PUBLISH_SM_QC",
        "PUBLISH_SM_TRIMMED_VID",
        "PUBLISH_SM_V6_FEATURES",
        "SELECT_COLUMNS",
        "URLIFY_FILE",
        "VIDEO_TO_POSE",
    ):
        continue
    model = LinearRegression()
    _ = model.fit(
        df_subset["video_duration_min"].to_numpy().reshape(-1, 1),
        df_subset["value"].to_numpy().reshape(-1, 1),
    )
    plot = (
        p9.ggplot(df_subset, p9.aes(x="video_duration_min", y="value"))
        + p9.geom_point()
        + p9.geom_smooth(method="lm")
        + p9.geom_text(
            p9.aes(x="x", y="y", label="label"),
            data=pd.DataFrame(
                {
                    "label": [
                        f"y = {np.round(model.coef_[0][0], 4)} * x + {np.round(model.intercept_[0], 4)}"
                    ],
                    "x": [df_subset["video_duration_min"].min() + 5],
                    "y": [df_subset["value"].max() * 1.2],
                }
            ),
        )
        + p9.theme_bw()
        + p9.labs(y=grp[1], x="Video duration (min)")
        + p9.ggtitle(f"Task: {grp[0]}")
    )
    plot.save(f"{grp[0]}_{grp[1]}.png", width=10, height=10, dpi=300)
