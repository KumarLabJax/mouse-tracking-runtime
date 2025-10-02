"""Pose data conversion utilities."""

import numpy as np

from mouse_tracking.utils.run_length_encode import run_length_encode


def v2_to_v3(pose_data, conf_data, threshold: float = 0.3):
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
    rle_starts, rle_durations, rle_values = run_length_encode(instance_count)
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


def multi_to_v2(pose_data, conf_data, identity_data):
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
