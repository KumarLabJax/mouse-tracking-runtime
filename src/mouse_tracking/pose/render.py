"""Renders pose data."""

import os

import cv2
import h5py
import imageio
import numpy as np

from mouse_tracking.core.config.pose_utils import PoseUtilsConfig
from mouse_tracking.pose import convert
from mouse_tracking.utils.segmentation import render_segmentation_overlay
from mouse_tracking.utils.static_objects import plot_keypoints

CONFIG = PoseUtilsConfig()


def render_pose_overlay(
    image: np.ndarray,
    frame_points: np.ndarray,
    exclude_points: list | None = None,
    color: tuple = (255, 255, 255),
) -> np.ndarray:
    """Renders a single pose on an image.

    Args:
            image: image to render pose on
            frame_points: keypoints to render. keypoints are ordered [y, x]
            exclude_points: set of keypoint indices to exclude
            color: color to render the pose

    Returns:
            modified image
    """
    if exclude_points is None:
        exclude_points = []

    new_image = image.copy()
    missing_keypoints = np.where(np.all(frame_points == 0, axis=-1))[0].tolist()
    exclude_points = set(exclude_points + missing_keypoints)

    def gen_line_fragments():
        """Created lines to draw."""
        for curr_pt_indexes in CONFIG.CONNECTED_SEGMENTS:
            curr_fragment = []
            for curr_pt_index in curr_pt_indexes:
                if curr_pt_index in exclude_points:
                    if len(curr_fragment) >= 2:
                        yield curr_fragment
                    curr_fragment = []
                else:
                    curr_fragment.append(curr_pt_index)
            if len(curr_fragment) >= 2:
                yield curr_fragment

    line_pt_indexes = list(gen_line_fragments())

    for curr_line_indexes in line_pt_indexes:
        line_pts = np.array(
            [(pt_x, pt_y) for pt_y, pt_x in frame_points[curr_line_indexes]], np.int32
        )
        if np.any(np.all(line_pts == 0, axis=-1)):
            continue
        cv2.polylines(new_image, [line_pts], False, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.polylines(new_image, [line_pts], False, color, 1, cv2.LINE_AA)

    for point_index in range(12):
        if point_index in exclude_points:
            continue
        point_y, point_x = frame_points[point_index, :]
        cv2.circle(new_image, (point_x, point_y), 3, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(new_image, (point_x, point_y), 2, color, -1, cv2.LINE_AA)

    return new_image


def process_video(
    in_video_path, pose_h5_path, out_video_path, disable_id: bool = False
):
    """Renders pose file related data onto a video.

    Args:
        in_video_path: input video
        pose_h5_path: input pose file
        out_video_path: output video
        disable_id: bool indicating to fall back to tracklet data (v3) instead of longterm id data (v4)

    Raises:
        FileNotFoundError if either input is missing.
    """
    if not os.path.isfile(in_video_path):
        raise FileNotFoundError(f"ERROR: missing file: {in_video_path}")
    if not os.path.isfile(pose_h5_path):
        raise FileNotFoundError(f"ERROR: missing file: {pose_h5_path}")
    # Read in all the necessary data
    with h5py.File(pose_h5_path, "r") as pose_h5:
        if "version" in pose_h5["poseest"].attrs:
            major_version = pose_h5["poseest"].attrs["version"][0]
        else:
            major_version = 2
        all_points = pose_h5["poseest/points"][:]
        # v6 stores segmentation data
        if major_version >= 6:
            all_seg_data = pose_h5["poseest/seg_data"][:]
            if not disable_id:
                all_seg_id = pose_h5["poseest/longterm_seg_id"][:]
            else:
                all_seg_id = pose_h5["poseest/instance_seg_id"][:]
        else:
            all_seg_data = None
            all_seg_id = None
        # v5 stores optional static object data.
        all_static_object_data = {}
        if major_version >= 5 and "static_objects" in pose_h5:
            for key in pose_h5["static_objects"]:
                all_static_object_data[key] = pose_h5[f"static_objects/{key}"][:]
        # v4 stores identity/tracklet merging data
        if major_version >= 4 and not disable_id:
            all_track_id = pose_h5["poseest/instance_embed_id"][:]
        elif major_version >= 3:
            all_track_id = pose_h5["poseest/instance_track_id"][:]
        # Data is v2, upgrade it to v3
        else:
            conf_data = pose_h5["poseest/confidence"][:]
            all_points, _, _, _, all_track_id = convert.v2_to_v3(all_points, conf_data)

    # Process the video
    with (
        imageio.get_reader(in_video_path) as video_reader,
        imageio.get_writer(out_video_path, fps=30) as video_writer,
    ):
        for frame_index, image in enumerate(video_reader):
            for obj_key, obj_data in all_static_object_data.items():
                # Arena corners are TL, TR, BL, BR, so sort them into a correct polygon for plotting
                # TODO: possibly use `sort_corners`?
                if obj_key == "corners":
                    obj_data = obj_data[[0, 1, 3, 2]]
                image = plot_keypoints(
                    obj_data,
                    image,
                    color=CONFIG.STATIC_OBJ_COLORS[obj_key],
                    is_yx=not CONFIG.STATIC_OBJ_XY[obj_key],
                    include_lines=obj_key != "lixit",
                )
            for pose_idx, pose_id in enumerate(all_track_id[frame_index]):
                image = render_pose_overlay(
                    image,
                    all_points[frame_index, pose_idx],
                    color=CONFIG.MOUSE_COLORS[pose_id % len(CONFIG.MOUSE_COLORS)],
                )
            if all_seg_data is not None:
                for seg_idx, seg_id in enumerate(all_seg_id[frame_index]):
                    image = render_segmentation_overlay(
                        all_seg_data[frame_index, seg_idx],
                        image,
                        color=CONFIG.MOUSE_COLORS[seg_id % len(CONFIG.MOUSE_COLORS)],
                    )
            video_writer.append_data(image)
    print(f"finished generating video: {out_video_path}", flush=True)
