"""Helper functions for rendering information on frames."""

import cv2
import numpy as np


def plot_frame_info(frame: np.ndarray, info_text: str):
    """
    Plots information on a video frame.

    Args:
            frame: The video frame to annotate
            info_text: The text to display on the frame

    Returns:
            Copy of frame with text overlay
    """
    # Get a copy of the frame to draw on
    annotated_frame = frame.copy()

    # Define the position for the text
    text_position = [25, 25]  # Top left

    # Put the text on the frame
    # Black bordered orange
    cv2.putText(
        annotated_frame,
        info_text,
        text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),  # Black
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated_frame,
        info_text,
        text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (241, 163, 64),  # Orange
        1,
        cv2.LINE_AA,
    )

    return annotated_frame
