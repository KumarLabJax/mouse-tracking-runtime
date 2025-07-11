"""Mouse Tracking Runtime inference CLI."""

from pathlib import Path
from typing import Annotated

import click
import typer

from mouse_tracking.pytorch_inference import (
    infer_fecal_boli_pytorch,
    infer_multi_pose_pytorch,
    infer_single_pose_pytorch,
)

# Import inference functions
from mouse_tracking.tfs_inference import (
    infer_arena_corner_model,
    infer_food_hopper_model,
    infer_lixit_model,
    infer_multi_identity_tfs,
    infer_multi_segmentation_tfs,
    infer_single_segmentation_tfs,
)

app = typer.Typer()


@app.command()
def arena_corner(
    video: Annotated[
        Path | None,
        typer.Option("--video", help="Video file for processing"),
    ] = None,
    frame: Annotated[
        Path | None,
        typer.Option("--frame", help="Image file for processing"),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Trained model to infer",
            click_type=click.Choice(["social-2022-pipeline"]),
        ),
    ] = "social-2022-pipeline",
    runtime: Annotated[
        str,
        typer.Option(
            "--runtime",
            help="Runtime to execute the model",
            click_type=click.Choice(["tfs"]),
        ),
    ] = "tfs",
    out_file: Annotated[
        Path | None,
        typer.Option("--out-file", help="Pose file to write out"),
    ] = None,
    out_image: Annotated[
        Path | None,
        typer.Option("--out-image", help="Render the final prediction to an image"),
    ] = None,
    out_video: Annotated[
        Path | None,
        typer.Option("--out-video", help="Render all predictions to a video"),
    ] = None,
    num_frames: Annotated[
        int, typer.Option("--num-frames", help="Number of frames to predict on")
    ] = 100,
    frame_interval: Annotated[
        int, typer.Option("--frame-interval", help="Interval of frames to predict on")
    ] = 100,
) -> None:
    """
    Infer arena corner detection model.

    Processes either a video file or a single frame image for arena corner detection.
    Exactly one of --video or --frame must be specified.

    Args:
        video: Path to video file for processing
        frame: Path to image file for processing
        model: Trained model to use for inference
        runtime: Runtime environment to execute the model
        out_file: Path to output pose file
        out_image: Path to render final prediction as image
        out_video: Path to render all predictions as video
        num_frames: Number of frames to predict on
        frame_interval: Interval of frames to predict on

    Raises:
        typer.Exit: If validation fails or file doesn't exist
    """
    # Validate mutually exclusive group
    if video and frame:
        typer.echo("Error: Cannot specify both --video and --frame options.", err=True)
        raise typer.Exit(1)

    if not video and not frame:
        typer.echo("Error: Must specify either --video or --frame option.", err=True)
        raise typer.Exit(1)

    # Determine input source and validate it exists
    input_source = video if video else frame
    if not input_source.exists():
        typer.echo(f"Error: Input file '{input_source}' does not exist.", err=True)
        raise typer.Exit(1)

    # Create args object compatible with existing inference function
    class InferenceArgs:
        """Arguments container for compatibility with existing inference code."""

        def __init__(self):
            self.model = model
            self.runtime = runtime
            self.video = str(video) if video else None
            self.frame = str(frame) if frame else None
            self.out_file = str(out_file) if out_file else None
            self.out_image = str(out_image) if out_image else None
            self.out_video = str(out_video) if out_video else None
            self.num_frames = num_frames
            self.frame_interval = frame_interval

    args = InferenceArgs()

    # Execute inference based on runtime
    if runtime == "tfs":
        infer_arena_corner_model(args)


@app.command()
def fecal_boli(
    video: Annotated[
        Path | None,
        typer.Option("--video", help="Video file for processing"),
    ] = None,
    frame: Annotated[
        Path | None,
        typer.Option("--frame", help="Image file for processing"),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Trained model to infer",
            click_type=click.Choice(["fecal-boli"]),
        ),
    ] = "fecal-boli",
    runtime: Annotated[
        str,
        typer.Option(
            "--runtime",
            help="Runtime to execute the model",
            click_type=click.Choice(["pytorch"]),
        ),
    ] = "pytorch",
    out_file: Annotated[
        Path | None,
        typer.Option("--out-file", help="Pose file to write out"),
    ] = None,
    out_image: Annotated[
        Path | None,
        typer.Option("--out-image", help="Render the final prediction to an image"),
    ] = None,
    out_video: Annotated[
        Path | None,
        typer.Option("--out-video", help="Render all predictions to a video"),
    ] = None,
    frame_interval: Annotated[
        int, typer.Option("--frame-interval", help="Interval of frames to predict on")
    ] = 1800,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Batch size to use while making predictions"),
    ] = 1,
) -> None:
    """
    Run fecal boli inference.

    Processes either a video file or a single frame image for fecal boli detection.
    Exactly one of --video or --frame must be specified.

    Args:
        video: Path to video file for processing
        frame: Path to image file for processing
        model: Trained model to use for inference
        runtime: Runtime environment to execute the model
        out_file: Path to output pose file
        out_image: Path to render final prediction as image
        out_video: Path to render all predictions as video
        frame_interval: Interval of frames to predict on
        batch_size: Batch size to use while making predictions

    Raises:
        typer.Exit: If validation fails or file doesn't exist
    """
    # Validate mutually exclusive group
    if video and frame:
        typer.echo("Error: Cannot specify both --video and --frame options.", err=True)
        raise typer.Exit(1)

    if not video and not frame:
        typer.echo("Error: Must specify either --video or --frame option.", err=True)
        raise typer.Exit(1)

    # Determine input source and validate it exists
    input_source = video if video else frame
    if not input_source.exists():
        typer.echo(f"Error: Input file '{input_source}' does not exist.", err=True)
        raise typer.Exit(1)

    # Create args object compatible with existing inference function
    class InferenceArgs:
        """Arguments container for compatibility with existing inference code."""

        def __init__(self):
            self.model = model
            self.runtime = runtime
            self.video = str(video) if video else None
            self.frame = str(frame) if frame else None
            self.out_file = str(out_file) if out_file else None
            self.out_image = str(out_image) if out_image else None
            self.out_video = str(out_video) if out_video else None
            self.frame_interval = frame_interval
            self.batch_size = batch_size

    args = InferenceArgs()

    # Execute inference based on runtime
    if runtime == "pytorch":
        infer_fecal_boli_pytorch(args)


@app.command()
def food_hopper(
    video: Annotated[
        Path | None,
        typer.Option("--video", help="Video file for processing"),
    ] = None,
    frame: Annotated[
        Path | None,
        typer.Option("--frame", help="Image file for processing"),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Trained model to infer",
            click_type=click.Choice(["social-2022-pipeline"]),
        ),
    ] = "social-2022-pipeline",
    runtime: Annotated[
        str,
        typer.Option(
            "--runtime",
            help="Runtime to execute the model",
            click_type=click.Choice(["tfs"]),
        ),
    ] = "tfs",
    out_file: Annotated[
        Path | None,
        typer.Option("--out-file", help="Pose file to write out"),
    ] = None,
    out_image: Annotated[
        Path | None,
        typer.Option("--out-image", help="Render the final prediction to an image"),
    ] = None,
    out_video: Annotated[
        Path | None,
        typer.Option("--out-video", help="Render all predictions to a video"),
    ] = None,
    num_frames: Annotated[
        int, typer.Option("--num-frames", help="Number of frames to predict on")
    ] = 100,
    frame_interval: Annotated[
        int, typer.Option("--frame-interval", help="Interval of frames to predict on")
    ] = 100,
) -> None:
    """
    Run food hopper inference.

    Processes either a video file or a single frame image for food hopper detection.
    Exactly one of --video or --frame must be specified.

    Args:
        video: Path to video file for processing
        frame: Path to image file for processing
        model: Trained model to use for inference
        runtime: Runtime environment to execute the model
        out_file: Path to output pose file
        out_image: Path to render final prediction as image
        out_video: Path to render all predictions as video
        num_frames: Number of frames to predict on
        frame_interval: Interval of frames to predict on

    Raises:
        typer.Exit: If validation fails or file doesn't exist
    """
    # Validate mutually exclusive group
    if video and frame:
        typer.echo("Error: Cannot specify both --video and --frame options.", err=True)
        raise typer.Exit(1)

    if not video and not frame:
        typer.echo("Error: Must specify either --video or --frame option.", err=True)
        raise typer.Exit(1)

    # Determine input source and validate it exists
    input_source = video if video else frame
    if not input_source.exists():
        typer.echo(f"Error: Input file '{input_source}' does not exist.", err=True)
        raise typer.Exit(1)

    # Create args object compatible with existing inference function
    class InferenceArgs:
        """Arguments container for compatibility with existing inference code."""

        def __init__(self):
            self.model = model
            self.runtime = runtime
            self.video = str(video) if video else None
            self.frame = str(frame) if frame else None
            self.out_file = str(out_file) if out_file else None
            self.out_image = str(out_image) if out_image else None
            self.out_video = str(out_video) if out_video else None
            self.num_frames = num_frames
            self.frame_interval = frame_interval

    args = InferenceArgs()

    # Execute inference based on runtime
    if runtime == "tfs":
        infer_food_hopper_model(args)


@app.command()
def lixit(
    video: Annotated[
        Path | None,
        typer.Option("--video", help="Video file for processing"),
    ] = None,
    frame: Annotated[
        Path | None,
        typer.Option("--frame", help="Image file for processing"),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Trained model to infer",
            click_type=click.Choice(["social-2022-pipeline"]),
        ),
    ] = "social-2022-pipeline",
    runtime: Annotated[
        str,
        typer.Option(
            "--runtime",
            help="Runtime to execute the model",
            click_type=click.Choice(["tfs"]),
        ),
    ] = "tfs",
    out_file: Annotated[
        Path | None,
        typer.Option("--out-file", help="Pose file to write out"),
    ] = None,
    out_image: Annotated[
        Path | None,
        typer.Option("--out-image", help="Render the final prediction to an image"),
    ] = None,
    out_video: Annotated[
        Path | None,
        typer.Option("--out-video", help="Render all predictions to a video"),
    ] = None,
    num_frames: Annotated[
        int, typer.Option("--num-frames", help="Number of frames to predict on")
    ] = 100,
    frame_interval: Annotated[
        int, typer.Option("--frame-interval", help="Interval of frames to predict on")
    ] = 100,
) -> None:
    """
    Run lixit inference.

    Processes either a video file or a single frame image for lixit water spout detection.
    Exactly one of --video or --frame must be specified.

    Args:
        video: Path to video file for processing
        frame: Path to image file for processing
        model: Trained model to use for inference
        runtime: Runtime environment to execute the model
        out_file: Path to output pose file
        out_image: Path to render final prediction as image
        out_video: Path to render all predictions as video
        num_frames: Number of frames to predict on
        frame_interval: Interval of frames to predict on

    Raises:
        typer.Exit: If validation fails or file doesn't exist
    """
    # Validate mutually exclusive group
    if video and frame:
        typer.echo("Error: Cannot specify both --video and --frame options.", err=True)
        raise typer.Exit(1)

    if not video and not frame:
        typer.echo("Error: Must specify either --video or --frame option.", err=True)
        raise typer.Exit(1)

    # Determine input source and validate it exists
    input_source = video if video else frame
    if not input_source.exists():
        typer.echo(f"Error: Input file '{input_source}' does not exist.", err=True)
        raise typer.Exit(1)

    # Create args object compatible with existing inference function
    class InferenceArgs:
        """Arguments container for compatibility with existing inference code."""

        def __init__(self):
            self.model = model
            self.runtime = runtime
            self.video = str(video) if video else None
            self.frame = str(frame) if frame else None
            self.out_file = str(out_file) if out_file else None
            self.out_image = str(out_image) if out_image else None
            self.out_video = str(out_video) if out_video else None
            self.num_frames = num_frames
            self.frame_interval = frame_interval

    args = InferenceArgs()

    # Execute inference based on runtime
    if runtime == "tfs":
        infer_lixit_model(args)


@app.command()
def multi_identity(
    out_file: Annotated[
        Path,
        typer.Option("--out-file", help="Pose file to write out"),
    ],
    video: Annotated[
        Path | None,
        typer.Option("--video", help="Video file for processing"),
    ] = None,
    frame: Annotated[
        Path | None,
        typer.Option("--frame", help="Image file for processing"),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Trained model to infer",
            click_type=click.Choice(["social-paper", "2023"]),
        ),
    ] = "social-paper",
    runtime: Annotated[
        str,
        typer.Option(
            "--runtime",
            help="Runtime to execute the model",
            click_type=click.Choice(["tfs"]),
        ),
    ] = "tfs",
) -> None:
    """
    Run multi-identity inference.

    Processes either a video file or a single frame image for mouse identity detection.
    Exactly one of --video or --frame must be specified.

    Args:
        out_file: Path to output pose file (required)
        video: Path to video file for processing
        frame: Path to image file for processing
        model: Trained model to use for inference
        runtime: Runtime environment to execute the model

    Raises:
        typer.Exit: If validation fails or file doesn't exist
    """
    # Validate mutually exclusive group
    if video and frame:
        typer.echo("Error: Cannot specify both --video and --frame options.", err=True)
        raise typer.Exit(1)

    if not video and not frame:
        typer.echo("Error: Must specify either --video or --frame option.", err=True)
        raise typer.Exit(1)

    # Determine input source and validate it exists
    input_source = video if video else frame
    if not input_source.exists():
        typer.echo(f"Error: Input file '{input_source}' does not exist.", err=True)
        raise typer.Exit(1)

    # Create args object compatible with existing inference function
    class InferenceArgs:
        """Arguments container for compatibility with existing inference code."""

        def __init__(self):
            self.model = model
            self.runtime = runtime
            self.video = str(video) if video else None
            self.frame = str(frame) if frame else None
            self.out_file = str(out_file)

    args = InferenceArgs()

    # Execute inference based on runtime
    if runtime == "tfs":
        infer_multi_identity_tfs(args)


@app.command()
def multi_pose(
    out_file: Annotated[
        Path,
        typer.Option("--out-file", help="Pose file to write out"),
    ],
    video: Annotated[
        Path | None,
        typer.Option("--video", help="Video file for processing"),
    ] = None,
    frame: Annotated[
        Path | None,
        typer.Option("--frame", help="Image file for processing"),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Trained model to infer",
            click_type=click.Choice(["social-paper-topdown"]),
        ),
    ] = "social-paper-topdown",
    runtime: Annotated[
        str,
        typer.Option(
            "--runtime",
            help="Runtime to execute the model",
            click_type=click.Choice(["pytorch"]),
        ),
    ] = "pytorch",
    out_video: Annotated[
        Path | None,
        typer.Option("--out-video", help="Render the results to a video"),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Batch size to use while making predictions"),
    ] = 1,
) -> None:
    """
    Run multi-pose inference.

    Processes either a video file or a single frame image for multi-mouse pose detection.
    Exactly one of --video or --frame must be specified.

    Args:
        out_file: Path to output pose file (required)
        video: Path to video file for processing
        frame: Path to image file for processing
        model: Trained model to use for inference
        runtime: Runtime environment to execute the model
        out_video: Path to render results as video
        batch_size: Batch size to use while making predictions

    Raises:
        typer.Exit: If validation fails or file doesn't exist
    """
    # Validate mutually exclusive group
    if video and frame:
        typer.echo("Error: Cannot specify both --video and --frame options.", err=True)
        raise typer.Exit(1)

    if not video and not frame:
        typer.echo("Error: Must specify either --video or --frame option.", err=True)
        raise typer.Exit(1)

    # Determine input source and validate it exists
    input_source = video if video else frame
    if not input_source.exists():
        typer.echo(f"Error: Input file '{input_source}' does not exist.", err=True)
        raise typer.Exit(1)

    # Validate that out_file exists (required for multi_pose)
    if not out_file.exists():
        typer.echo(
            f"Error: Pose file containing segmentation data is required. Pose file '{out_file}' does not exist.",
            err=True,
        )
        raise typer.Exit(1)

    # Create args object compatible with existing inference function
    class InferenceArgs:
        """Arguments container for compatibility with existing inference code."""

        def __init__(self):
            self.model = model
            self.runtime = runtime
            self.video = str(video) if video else None
            self.frame = str(frame) if frame else None
            self.out_file = str(out_file)
            self.out_video = str(out_video) if out_video else None
            self.batch_size = batch_size

    args = InferenceArgs()

    # Execute inference based on runtime
    if runtime == "pytorch":
        infer_multi_pose_pytorch(args)


@app.command()
def single_pose(
    out_file: Annotated[
        Path,
        typer.Option("--out-file", help="Pose file to write out"),
    ],
    video: Annotated[
        Path | None,
        typer.Option("--video", help="Video file for processing"),
    ] = None,
    frame: Annotated[
        Path | None,
        typer.Option("--frame", help="Image file for processing"),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Trained model to infer",
            click_type=click.Choice(["gait-paper"]),
        ),
    ] = "gait-paper",
    runtime: Annotated[
        str,
        typer.Option(
            "--runtime",
            help="Runtime to execute the model",
            click_type=click.Choice(["pytorch"]),
        ),
    ] = "pytorch",
    out_video: Annotated[
        Path | None,
        typer.Option("--out-video", help="Render the results to a video"),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Batch size to use while making predictions"),
    ] = 1,
) -> None:
    """
    Run single-pose inference.

    Processes either a video file or a single frame image for single-mouse pose detection.
    Exactly one of --video or --frame must be specified.

    Args:
        out_file: Path to output pose file (required)
        video: Path to video file for processing
        frame: Path to image file for processing
        model: Trained model to use for inference
        runtime: Runtime environment to execute the model
        out_video: Path to render results as video
        batch_size: Batch size to use while making predictions

    Raises:
        typer.Exit: If validation fails or file doesn't exist
    """
    # Validate mutually exclusive group
    if video and frame:
        typer.echo("Error: Cannot specify both --video and --frame options.", err=True)
        raise typer.Exit(1)

    if not video and not frame:
        typer.echo("Error: Must specify either --video or --frame option.", err=True)
        raise typer.Exit(1)

    # Determine input source and validate it exists
    input_source = video if video else frame
    if not input_source.exists():
        typer.echo(f"Error: Input file '{input_source}' does not exist.", err=True)
        raise typer.Exit(1)

    # Create args object compatible with existing inference function
    class InferenceArgs:
        """Arguments container for compatibility with existing inference code."""

        def __init__(self):
            self.model = model
            self.runtime = runtime
            self.video = str(video) if video else None
            self.frame = str(frame) if frame else None
            self.out_file = str(out_file)
            self.out_video = str(out_video) if out_video else None
            self.batch_size = batch_size

    args = InferenceArgs()

    # Execute inference based on runtime
    if runtime == "pytorch":
        infer_single_pose_pytorch(args)


@app.command()
def single_segmentation(
    out_file: Annotated[
        Path,
        typer.Option("--out-file", help="Pose file to write out"),
    ],
    video: Annotated[
        Path | None,
        typer.Option("--video", help="Video file for processing"),
    ] = None,
    frame: Annotated[
        Path | None,
        typer.Option("--frame", help="Image file for processing"),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Trained model to infer",
            click_type=click.Choice(["tracking-paper"]),
        ),
    ] = "tracking-paper",
    runtime: Annotated[
        str,
        typer.Option(
            "--runtime",
            help="Runtime to execute the model",
            click_type=click.Choice(["tfs"]),
        ),
    ] = "tfs",
    out_video: Annotated[
        Path | None,
        typer.Option("--out-video", help="Render the results to a video"),
    ] = None,
) -> None:
    """
    Run single-segmentation inference.

    Processes either a video file or a single frame image for single-mouse segmentation.
    Exactly one of --video or --frame must be specified.

    Args:
        out_file: Path to output pose file (required)
        video: Path to video file for processing
        frame: Path to image file for processing
        model: Trained model to use for inference
        runtime: Runtime environment to execute the model
        out_video: Path to render results as video

    Raises:
        typer.Exit: If validation fails or file doesn't exist
    """
    # Validate mutually exclusive group
    if video and frame:
        typer.echo("Error: Cannot specify both --video and --frame options.", err=True)
        raise typer.Exit(1)

    if not video and not frame:
        typer.echo("Error: Must specify either --video or --frame option.", err=True)
        raise typer.Exit(1)

    # Determine input source and validate it exists
    input_source = video if video else frame
    if not input_source.exists():
        typer.echo(f"Error: Input file '{input_source}' does not exist.", err=True)
        raise typer.Exit(1)

    # Create args object compatible with existing inference function
    class InferenceArgs:
        """Arguments container for compatibility with existing inference code."""

        def __init__(self):
            self.model = model
            self.runtime = runtime
            self.video = str(video) if video else None
            self.frame = str(frame) if frame else None
            self.out_file = str(out_file)
            self.out_video = str(out_video) if out_video else None

    args = InferenceArgs()

    # Execute inference based on runtime
    if runtime == "tfs":
        infer_single_segmentation_tfs(args)


# Add multi_segmentation command that was missing
@app.command()
def multi_segmentation(
    out_file: Annotated[
        Path,
        typer.Option("--out-file", help="Pose file to write out"),
    ],
    video: Annotated[
        Path | None,
        typer.Option("--video", help="Video file for processing"),
    ] = None,
    frame: Annotated[
        Path | None,
        typer.Option("--frame", help="Image file for processing"),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Trained model to infer",
            click_type=click.Choice(["social-paper"]),
        ),
    ] = "social-paper",
    runtime: Annotated[
        str,
        typer.Option(
            "--runtime",
            help="Runtime to execute the model",
            click_type=click.Choice(["tfs"]),
        ),
    ] = "tfs",
    out_video: Annotated[
        Path | None,
        typer.Option("--out-video", help="Render the results to a video"),
    ] = None,
) -> None:
    """
    Run multi-segmentation inference.

    Processes either a video file or a single frame image for multi-mouse segmentation.
    Exactly one of --video or --frame must be specified.

    Args:
        out_file: Path to output pose file (required)
        video: Path to video file for processing
        frame: Path to image file for processing
        model: Trained model to use for inference
        runtime: Runtime environment to execute the model
        out_video: Path to render results as video

    Raises:
        typer.Exit: If validation fails or file doesn't exist
    """
    # Validate mutually exclusive group
    if video and frame:
        typer.echo("Error: Cannot specify both --video and --frame options.", err=True)
        raise typer.Exit(1)

    if not video and not frame:
        typer.echo("Error: Must specify either --video or --frame option.", err=True)
        raise typer.Exit(1)

    # Determine input source and validate it exists
    input_source = video if video else frame
    if not input_source.exists():
        typer.echo(f"Error: Input file '{input_source}' does not exist.", err=True)
        raise typer.Exit(1)

    # Create args object compatible with existing inference function
    class InferenceArgs:
        """Arguments container for compatibility with existing inference code."""

        def __init__(self):
            self.model = model
            self.runtime = runtime
            self.video = str(video) if video else None
            self.frame = str(frame) if frame else None
            self.out_file = str(out_file)
            self.out_video = str(out_video) if out_video else None

    args = InferenceArgs()

    # Execute inference based on runtime
    if runtime == "tfs":
        infer_multi_segmentation_tfs(args)
