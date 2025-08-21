"""Helper utilities for the CLI."""

from pathlib import Path

import typer
from rich import print

from mouse_tracking import __version__
from mouse_tracking.matching.match_predictions import match_predictions
from mouse_tracking.pose import render
from mouse_tracking.pose.convert import downgrade_pose_file
from mouse_tracking.utils import fecal_boli, static_objects
from mouse_tracking.utils.clip_video import clip_video_auto, clip_video_manual

app = typer.Typer()


def version_callback(value: bool) -> None:
    """
    Display the application version and exit.

    Args:
        value: Flag indicating whether to show version

    """
    if value:
        print(f"Mouse Tracking Runtime version: [green]{__version__}[/green]")
        raise typer.Exit()


@app.command()
def aggregate_fecal_boli(
    folder: Path = typer.Argument(
        ..., help="Path to the folder containing fecal boli data"
    ),
    folder_depth: int = typer.Option(
        2, help="Expected subfolder depth in the project folder"
    ),
    num_bins: int = typer.Option(
        -1, help="Number of bins to read in (value < 0 reads all)"
    ),
    output: Path = typer.Option(
        "output.csv", help="Output file path for aggregated data"
    ),
):
    """
    Aggregate fecal boli data.

    This command processes and aggregates fecal boli data from the specified source.
    """
    result = fecal_boli.aggregate_folder_data(
        str(folder), depth=folder_depth, num_bins=num_bins
    )
    result.to_csv(output, index=False)


clip_video_app = typer.Typer(help="Produce a video and pose clip aligned to criteria.")


@clip_video_app.command()
def auto(
    in_video: str = typer.Option(..., "--in-video", help="input video file"),
    in_pose: str = typer.Option(..., "--in-pose", help="input HDF5 pose file"),
    out_video: str = typer.Option(..., "--out-video", help="output video file"),
    out_pose: str = typer.Option(..., "--out-pose", help="output HDF5 pose file"),
    allow_overwrite: bool = typer.Option(
        False,
        "--allow-overwrite",
        help="Allows existing files to be overwritten (default error)",
    ),
    observation_duration: int = typer.Option(
        30 * 60 * 60,
        "--observation-duration",
        help="Duration of the observation to clip. (Default 1hr)",
    ),
    frame_offset: int = typer.Option(
        150,
        "--frame-offset",
        help="Number of frames to offset from the first detected pose. Positive values indicate adding time before. (Default 150)",
    ),
    num_keypoints: int = typer.Option(
        12,
        "--num-keypoints",
        help="Number of keypoints to consider a detected pose. (Default 12)",
    ),
    confidence_threshold: float = typer.Option(
        0.3,
        "--confidence-threshold",
        help="Minimum confidence of a keypoint to be considered valid. (Default 0.3)",
    ),
):
    """Automatically detect the first frame based on pose."""
    if not allow_overwrite:
        if Path(out_video).exists():
            msg = f"{out_video} exists. If you wish to overwrite, please include --allow-overwrite"
            raise FileExistsError(msg)
        if Path(out_pose).exists():
            msg = f"{out_pose} exists. If you wish to overwrite, please include --allow-overwrite"
            raise FileExistsError(msg)
    clip_video_auto(
        in_video,
        in_pose,
        out_video,
        out_pose,
        frame_offset=frame_offset,
        observation_duration=observation_duration,
        confidence_threshold=confidence_threshold,
        num_keypoints=num_keypoints,
    )


@clip_video_app.command()
def manual(
    in_video: str = typer.Option(..., "--in-video", help="input video file"),
    in_pose: str = typer.Option(..., "--in-pose", help="input HDF5 pose file"),
    out_video: str = typer.Option(..., "--out-video", help="output video file"),
    out_pose: str = typer.Option(..., "--out-pose", help="output HDF5 pose file"),
    allow_overwrite: bool = typer.Option(
        False,
        "--allow-overwrite",
        help="Allows existing files to be overwritten (default error)",
    ),
    observation_duration: int = typer.Option(
        30 * 60 * 60,
        "--observation-duration",
        help="Duration of the observation to clip. (Default 1hr)",
    ),
    frame_start: int = typer.Option(
        ..., "--frame-start", help="Frame to start the clip at"
    ),
):
    """Manually set the first frame."""
    if not allow_overwrite:
        if Path(out_video).exists():
            msg = f"{out_video} exists. If you wish to overwrite, please include --allow-overwrite"
            raise FileExistsError(msg)
        if Path(out_pose).exists():
            msg = f"{out_pose} exists. If you wish to overwrite, please include --allow-overwrite"
            raise FileExistsError(msg)

    clip_video_manual(
        in_video,
        in_pose,
        out_video,
        out_pose,
        frame_start,
        observation_duration=observation_duration,
    )


app.add_typer(
    clip_video_app,
    name="clip-video-to-start",
    help="Clip video and pose data based on specified criteria",
)


@app.command()
def downgrade_multi_to_single(
    in_pose: Path = typer.Argument(..., help="Input HDF5 pose file path"),
    disable_id: bool = typer.Option(
        False,
        "--disable-id",
        help="Disable identity embedding tracks (if available) and use tracklet data instead",
    ),
):
    """
    Downgrade multi-identity data to single-identity.

    This command processes multi-identity data and downgrades it to single-identity format.
    """
    typer.echo(
        "Warning: Not all pipelines may be 100% compatible using downgraded pose"
        " files. Files produced from this script will contain 0s in data where "
        "low confidence predictions were made instead of the original values "
        "which may affect performance."
    )
    downgrade_pose_file(str(in_pose), disable_id=disable_id)


@app.command()
def flip_xy_field(
    in_pose: Path = typer.Argument(..., help="Input HDF5 pose file"),
    object_key: str = typer.Argument(
        ..., help="Data key to swap the sorting of [y, x] data to [x, y]"
    ),
):
    """
    Flip XY field.

    This command flips the XY coordinates in the dataset.
    """
    static_objects.swap_static_obj_xy(in_pose, object_key)


@app.command()
def render_pose(
    in_video: Path = typer.Argument(..., help="Input video file path"),
    in_pose: Path = typer.Argument(..., help="Input HDF5 pose file path"),
    out_video: Path = typer.Argument(..., help="Output video file path"),
    disable_id: bool = typer.Option(
        False,
        "--disable-id",
        help="Disable identity rendering (v4) and use track ids (v3) instead",
    ),
):
    """
    Render pose data.

    This command renders the pose data from the specified source.
    """
    render.process_video(
        str(in_video),
        str(in_pose),
        str(out_video),
        disable_id=disable_id,
    )


@app.command()
def stitch_tracklets(
    in_pose: Path = typer.Argument(..., help="Input HDF5 pose file"),
):
    """
    Stitch tracklets.

    This command stitches tracklets from the specified source.
    """
    match_predictions(in_pose)
