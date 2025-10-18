"""Helper functions for plotting data."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9

from mouse_tracking.utils.features import JABSFeature


def plot_jabs_feature(feature_file: Path, feature: str):
    """
    Generates a plot for a JABS feature for the requested time.

    Args:
        feature_file: JABS feature file
        feature: JABS feature key to plot

    Returns:
        matplotlib.figure.Figure of the plot

    Raises:
        ValueError when feature does not exist in the feature file.
    """
    feature_obj = JABSFeature(feature_file)
    if feature not in np.asarray(feature_obj.feature_keys[["key"]].values).reshape(
        [-1]
    ):
        raise ValueError(
            f"Feature {feature} not present in feature file {feature_file.name}"
        )

    feature_arr = feature_obj.get_key_data(feature)
    plot = (
        p9.ggplot(
            pd.DataFrame({"frame": np.arange(len(feature_arr)), "val": feature_arr}),
            p9.aes(x="frame/30/60", y="val"),
        )
        + p9.geom_point()
        + p9.theme_bw()
        + p9.labs(x="minute", y=feature)
    )

    return plot.draw()
