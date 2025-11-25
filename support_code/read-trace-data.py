#!/usr/bin/env python3
"""Generate plots using nextflow report data.

This script provides a template for reading data from a nextflow report html
and generating some additional plots on top of what nextflow provides.
"""

import json
import re
from pathlib import Path

import pandas as pd
import plotnine as p9
from bs4 import BeautifulSoup


def extract_nextflow_df_from_report(report_file):
    """
    Parses trace data from a nextflow file.

    Args:
        report_file: nextflow html file to parse

    Returns:
        dataframe of report data

    Todo:
        This function was written on a specific type of report output from nextflow when
        run with `--with-report` and `--with-trace`. Columns are hard-coded and
        may not be present in all reports generated.
        Times were not checked to be properly aligned to local time.
    """
    with open(report_file) as f:
        html_content = f.read()
    # Nextflow places data inside scripts
    soup = BeautifulSoup(html_content, "html.parser")
    script = soup.find("script")
    if len(script) == 0:
        raise ValueError("html report did not contain a scripts block with data.")
    # Nextflow trace data is titles 'window.data'
    table_data = script.getText("window.data")
    # This is a lazy json as raw text being assigned
    raw = table_data[re.search("window.data = ", table_data).span()[0] + 14 : -5]
    fixed = re.sub(r'\\(?![\\/"bfnrtu])', r"\\\\", raw)
    data = json.loads(fixed)
    df = pd.DataFrame(data["trace"])
    # Coercing string data into actually useful pandas types
    df["write_bytes"] = pd.to_numeric(df["write_bytes"], errors="coerce")
    df["read_bytes"] = pd.to_numeric(df["read_bytes"], errors="coerce")
    df["%cpu"] = pd.to_numeric(df["%cpu"], errors="coerce")
    df["%mem"] = pd.to_numeric(df["%mem"], errors="coerce")
    df["rss"] = pd.to_numeric(df["rss"], errors="coerce")
    df["vmem"] = pd.to_numeric(df["vmem"], errors="coerce")
    df["peak_rss"] = pd.to_numeric(df["peak_rss"], errors="coerce")
    df["peak_vmem"] = pd.to_numeric(df["peak_vmem"], errors="coerce")
    df["rchar"] = pd.to_numeric(df["rchar"], errors="coerce")
    df["wchar"] = pd.to_numeric(df["wchar"], errors="coerce")
    df["syscr"] = pd.to_numeric(df["syscr"], errors="coerce")
    df["syscw"] = pd.to_numeric(df["syscw"], errors="coerce")
    df["cpus"] = pd.to_numeric(df["cpus"], errors="coerce")
    df["memory"] = pd.to_numeric(df["memory"], errors="coerce")
    df["attempt"] = pd.to_numeric(df["attempt"], errors="coerce")
    df["vol_ctxt"] = pd.to_numeric(df["vol_ctxt"], errors="coerce")
    df["inv_ctxt"] = pd.to_numeric(df["inv_ctxt"], errors="coerce")
    df["time"] = pd.to_timedelta(pd.to_numeric(df["time"], errors="coerce"), unit="ms")
    df["time"] = pd.to_timedelta(
        pd.to_numeric(df["realtime"], errors="coerce"), unit="ms"
    )
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df["submit"] = pd.to_datetime(
        pd.to_numeric(df["submit"], errors="coerce"), unit="ms"
    )
    df.loc[df["submit"] < "1970-01-02", "submit"] = pd.NaT
    df["start"] = pd.to_datetime(pd.to_numeric(df["start"], errors="coerce"), unit="ms")
    df.loc[df["start"] < "1970-01-02", "start"] = pd.NaT
    df["complete"] = pd.to_datetime(
        pd.to_numeric(df["complete"], errors="coerce"), unit="ms"
    )
    df.loc[df["complete"] < "1970-01-02", "complete"] = pd.NaT
    df["duration"] = pd.to_timedelta(
        pd.to_numeric(df["duration"], errors="coerce"), unit="ms"
    )
    df["duration_s"] = df["duration"].dt.total_seconds()
    return df


# Basic loop to read in all reports in a folder.
dir_with_reports = (
    "/media/bgeuther/LL3_Internal/test-data/compression/to_cloudian/reports/"
)
dfs = []
for report_to_read in [
    x for x in Path(dir_with_reports).iterdir() if x.suffix == ".html"
]:
    dfs.append(extract_nextflow_df_from_report(report_to_read))

df = pd.concat(dfs)

# Some additional information added into the dataframe that nextflow doesn't log
# Add in the info for when the switch to 4 jobs was made
switch_date = "2025-11-14 8:00:00"
df["simultaneous_retrieval"] = 2
df.loc[df["start"] > switch_date, "simultaneous_retrieval"] = 4

# Some plots used to better understand some metrics

# Do retrieval speed tasks vary over the course of the week?
(
    p9.ggplot(
        df[df["process"] == "GET_DATA_FROM_DROPBOX"],
        p9.aes(x="start", y="write_bytes/duration_s", color="simultaneous_retrieval"),
    )
    + p9.geom_point()
    + p9.scale_y_log10()
    + p9.scale_x_datetime(date_labels="%Y-%m-%d %H:%M:%S")
    + p9.labs(title="Retrieval speed over day of week")
    + p9.theme_bw()
)

# What is the type of cluster loading between different types of jobs? (What could be a bottleneck?)
(
    p9.ggplot(df, p9.aes(x="start", y="process", color="process"))
    + p9.geom_point()
    + p9.scale_x_datetime(date_labels="%Y-%m-%d %H:%M:%S")
    + p9.labs(title="Job submit time over day of week")
    + p9.theme_bw()
)
