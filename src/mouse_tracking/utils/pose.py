import re
from pathlib import Path

import cv2
import h5py
import numpy as np

from mouse_tracking.utils.arrays import safe_find_first
from mouse_tracking.utils.hashing import hash_file
from mouse_tracking.utils.run_length_encode import rle

NOSE_INDEX = 0
LEFT_EAR_INDEX = 1
RIGHT_EAR_INDEX = 2
BASE_NECK_INDEX = 3
LEFT_FRONT_PAW_INDEX = 4
RIGHT_FRONT_PAW_INDEX = 5
CENTER_SPINE_INDEX = 6
LEFT_REAR_PAW_INDEX = 7
RIGHT_REAR_PAW_INDEX = 8
BASE_TAIL_INDEX = 9
MID_TAIL_INDEX = 10
TIP_TAIL_INDEX = 11

CONNECTED_SEGMENTS = [
    [LEFT_FRONT_PAW_INDEX, CENTER_SPINE_INDEX, RIGHT_FRONT_PAW_INDEX],
    [LEFT_REAR_PAW_INDEX, BASE_TAIL_INDEX, RIGHT_REAR_PAW_INDEX],
    [
        NOSE_INDEX,
        BASE_NECK_INDEX,
        CENTER_SPINE_INDEX,
        BASE_TAIL_INDEX,
        MID_TAIL_INDEX,
        TIP_TAIL_INDEX,
    ],
]

MIN_HIGH_CONFIDENCE = 0.75
MIN_GAIT_CONFIDENCE = 0.3
MIN_JABS_CONFIDENCE = 0.3
MIN_JABS_KEYPOINTS = 3


def convert_v2_to_v3(pose_data, conf_data, threshold: float = 0.3):
    """Converts single mouse pose data into multimouse.

    Args:
            pose_data: single mouse pose data of shape [frame, 12, 2]
            conf_data: keypoint confidence data of shape [frame, 12]
            threshold: threshold for filtering valid keypoint predictions
                    0.3 is used in JABS
                    0.4 is used for multi-mouse prediction code
                    0.5 is a typical default in other software

    Returns:
            tuple of (pose_data_v3, conf_data_v3, instance_count, instance_embedding, instance_track_id)
            pose_data_v3: pose_data reformatted to v3
            conf_data_v3: conf_data reformatted to v3
            instance_count: instance count field for v3 files
            instance_embedding: dummy data for embedding data field in v3 files
            instance_track_id: tracklet data for v3 files
    """
    pose_data_v3 = np.reshape(pose_data, [-1, 1, 12, 2])
    conf_data_v3 = np.reshape(conf_data, [-1, 1, 12])
    bad_pose_data = conf_data_v3 < threshold
    pose_data_v3[np.repeat(np.expand_dims(bad_pose_data, -1), 2, axis=-1)] = 0
    conf_data_v3[bad_pose_data] = 0
    instance_count = np.full([pose_data_v3.shape[0]], 1, dtype=np.uint8)
    instance_count[np.all(bad_pose_data, axis=-1).reshape(-1)] = 0
    instance_embedding = np.full(conf_data_v3.shape, 0, dtype=np.float32)
    # Tracks can only be continuous blocks
    instance_track_id = np.full(pose_data_v3.shape[:2], 0, dtype=np.uint32)
    rle_starts, rle_durations, rle_values = rle(instance_count)
    for i, (start, duration) in enumerate(
        zip(rle_starts[rle_values == 1], rle_durations[rle_values == 1], strict=False)
    ):
        instance_track_id[start : start + duration] = i
    return (
        pose_data_v3,
        conf_data_v3,
        instance_count,
        instance_embedding,
        instance_track_id,
    )


def convert_multi_to_v2(pose_data, conf_data, identity_data):
    """Converts multi mouse pose data (v3+) into multiple single mouse (v2).

    Args:
            pose_data: multi mouse pose data of shape [frame, max_animals, 12, 2]
            conf_data: keypoint confidence data of shape [frame, max_animals, 12]
            identity_data: identity data which indicates animal indices of shape [frame, max_animals]

    Returns:
            list of tuples containing (id, pose_data_v2, conf_data_v2)
            id: tracklet id
            pose_data_v2: pose_data reformatted to v2
            conf_data_v2: conf_data reformatted to v2

    Raises:
            ValueError if an identity has 2 pose predictions in a single frame.
    """
    invalid_poses = np.all(conf_data == 0, axis=-1)
    id_values = np.unique(identity_data[~invalid_poses])
    masked_id_data = identity_data.copy().astype(np.int32)
    # This is to handle id 0 (with 0-padding). -1 is an invalid id.
    masked_id_data[invalid_poses] = -1

    return_list = []
    for cur_id in id_values:
        id_frames, id_idxs = np.where(masked_id_data == cur_id)
        if len(id_frames) != len(set(id_frames)):
            sorted_frames = np.sort(id_frames)
            duplicated_frames = sorted_frames[:-1][
                sorted_frames[1:] == sorted_frames[:-1]
            ]
            msg = f"Identity {cur_id} contained multiple poses assigned on frames {duplicated_frames}."
            raise ValueError(msg)
        single_pose = np.zeros([len(pose_data), 12, 2], dtype=pose_data.dtype)
        single_conf = np.zeros([len(pose_data), 12], dtype=conf_data.dtype)
        single_pose[id_frames] = pose_data[id_frames, id_idxs]
        single_conf[id_frames] = conf_data[id_frames, id_idxs]

        return_list.append((cur_id, single_pose, single_conf))

    return return_list


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
        for curr_pt_indexes in CONNECTED_SEGMENTS:
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


def find_first_pose(
    confidence, confidence_threshold: float = 0.3, num_keypoints: int = 12
):
    """Detects the first pose with all the keypoints.

    Args:
            confidence: confidence matrix
            confidence_threshold: minimum confidence to be considered a valid keypoint. See `convert_v2_to_v3` for additional notes on confidences
            num_keypoints: number of keypoints

    Returns:
            integer indicating the first frame when the pose was observed.
            In the case of multi-animal, the first frame when any full pose was found

    Raises:
            ValueError if no pose meets the criteria
    """
    valid_keypoints = confidence > confidence_threshold
    num_keypoints_in_pose = np.sum(valid_keypoints, axis=-1)
    # Multi-mouse
    if num_keypoints_in_pose.ndim == 2:
        num_keypoints_in_pose = np.max(num_keypoints_in_pose, axis=-1)

    completed_pose_frames = np.argwhere(num_keypoints_in_pose >= num_keypoints)
    if len(completed_pose_frames) == 0:
        msg = f"No poses detected with {num_keypoints} keypoints and confidence threshold {confidence_threshold}"
        raise ValueError(msg)

    return completed_pose_frames[0][0]


def find_first_pose_file(
    pose_file, confidence_threshold: float = 0.3, num_keypoints: int = 12
):
    """Lazy wrapper for `find_first_pose` that reads in file data.

    Args:
            pose_file: pose file to read confidence matrix from
            confidence_threshold: see `find_first_pose`
            num_keypoints: see `find_first_pose`

    Returns:
            see `find_first_pose`
    """
    with h5py.File(pose_file, "r") as f:
        confidences = f["poseest/confidence"][...]

    return find_first_pose(confidences, confidence_threshold, num_keypoints)


def inspect_pose_v2(pose_file, pad: int = 150, duration: int = 108000):
    """Inspects a single mouse pose file v2 for coverage metrics.

    Args:
            pose_file: The pose file to inspect
            pad: pad size expected in the beginning
            duration: expected duration of experiment

    Returns:
            Dict containing the following keyed data:
                    first_frame_pose: First frame where the pose data appeared
                    first_frame_full_high_conf: First frame with 12 keypoints at high confidence
                    pose_counts: total number of poses predicted
                    missing_poses: missing poses in the primary duration of the video
                    missing_keypoint_frames: number of frames which don't contain 12 keypoints in the primary duration
    """
    with h5py.File(pose_file, "r") as f:
        pose_version = f["poseest"].attrs["version"][0]
        if pose_version != 2:
            msg = f"Only v2 pose files are supported for inspection. {pose_file} is version {pose_version}"
            raise ValueError(msg)
        pose_quality = f["poseest/confidence"][:]

    num_keypoints = np.sum(pose_quality > MIN_JABS_CONFIDENCE, axis=1)
    return_dict = {}
    return_dict["first_frame_pose"] = safe_find_first(np.all(num_keypoints, axis=1))
    high_conf_keypoints = np.all(pose_quality > MIN_HIGH_CONFIDENCE, axis=2).squeeze(1)
    return_dict["first_frame_full_high_conf"] = safe_find_first(high_conf_keypoints)
    return_dict["pose_counts"] = np.sum(num_keypoints > MIN_JABS_CONFIDENCE)
    return_dict["missing_poses"] = duration - np.sum(
        (num_keypoints > MIN_JABS_CONFIDENCE)[pad : pad + duration]
    )
    return_dict["missing_keypoint_frames"] = np.sum(
        num_keypoints[pad : pad + duration] != 12
    )
    return return_dict


def inspect_pose_v6(pose_file, pad: int = 150, duration: int = 108000):
    """Inspects a single mouse pose file v6 for coverage metrics.

    Args:
            pose_file: The pose file to inspect
            pad: duration of data skipped in the beginning (not observation period)
            duration: observation duration of experiment

    Returns:
            Dict containing the following keyed data:
                    pose_file: The pose file inspected
                    pose_hash: The blake2b hash of the pose file
                    video_name: The video name associated with the pose file (no extension)
                    video_duration: Duration of the video
                    corners_present: If the corners are present in the pose file
                    first_frame_pose: First frame where the pose data appeared
                    first_frame_full_high_conf: First frame with 12 keypoints > 0.75 confidence
                    first_frame_jabs: First frame with 3 keypoints > 0.3 confidence
                    first_frame_gait: First frame > 0.3 confidence for base tail and rear paws keypoints
                    first_frame_seg: First frame where segmentation data was assigned an id
                    pose_counts: Total number of poses predicted
                    seg_counts: Total number of segmentations matched with poses
                    missing_poses: Missing poses in the observation duration of the video
                    missing_segs: Missing segmentations in the observation duration of the video
                    pose_tracklets: Number of tracklets in the observation duration
                    missing_keypoint_frames: Number of frames which don't contain 12 keypoints in the observation duration
    """
    with h5py.File(pose_file, "r") as f:
        pose_version = f["poseest"].attrs["version"][0]
        if pose_version < 6:
            msg = f"Only v6+ pose files are supported for inspection. {pose_file} is version {pose_version}"
            raise ValueError(msg)
        pose_counts = f["poseest/instance_count"][:]
        if np.max(pose_counts) > 1:
            msg = f"Only single mouse pose files are supported for inspection. {pose_file} contains multiple instances"
            raise ValueError(msg)
        pose_quality = f["poseest/confidence"][:]
        pose_tracks = f["poseest/instance_track_id"][:]
        seg_ids = f["poseest/longterm_seg_id"][:]
        corners_present = "static_objects/corners" in f

    num_keypoints = 12 - np.sum(pose_quality.squeeze(1) == 0, axis=1)
    return_dict = {}
    return_dict["pose_file"] = Path(pose_file).name
    return_dict["pose_hash"] = hash_file(Path(pose_file))
    # Keep 2 folders if present for video name
    folder_name = "/".join(Path(pose_file).parts[-3:-1]) + "/"
    return_dict["video_name"] = folder_name + re.sub(
        "_pose_est_v[0-9]+", "", Path(pose_file).stem
    )
    return_dict["video_duration"] = pose_counts.shape[0]
    return_dict["corners_present"] = corners_present
    return_dict["first_frame_pose"] = safe_find_first(pose_counts > 0)
    high_conf_keypoints = np.all(pose_quality > MIN_HIGH_CONFIDENCE, axis=2).squeeze(1)
    return_dict["first_frame_full_high_conf"] = safe_find_first(high_conf_keypoints)
    jabs_keypoints = np.sum(pose_quality > MIN_JABS_CONFIDENCE, axis=2).squeeze(1)
    return_dict["first_frame_jabs"] = safe_find_first(
        jabs_keypoints >= MIN_JABS_KEYPOINTS
    )
    gait_keypoints = np.all(
        pose_quality[:, :, [BASE_TAIL_INDEX, LEFT_REAR_PAW_INDEX, RIGHT_REAR_PAW_INDEX]]
        > MIN_GAIT_CONFIDENCE,
        axis=2,
    ).squeeze(1)
    return_dict["first_frame_gait"] = safe_find_first(gait_keypoints)
    return_dict["first_frame_seg"] = safe_find_first(seg_ids > 0)
    return_dict["pose_counts"] = np.sum(pose_counts)
    return_dict["seg_counts"] = np.sum(seg_ids > 0)
    return_dict["missing_poses"] = duration - np.sum(pose_counts[pad : pad + duration])
    return_dict["missing_segs"] = duration - np.sum(seg_ids[pad : pad + duration] > 0)
    return_dict["pose_tracklets"] = len(
        np.unique(
            pose_tracks[pad : pad + duration][pose_counts[pad : pad + duration] == 1]
        )
    )
    return_dict["missing_keypoint_frames"] = np.sum(
        num_keypoints[pad : pad + duration] != 12
    )
    return return_dict
