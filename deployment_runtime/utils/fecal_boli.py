import numpy as np
import cv2
from typing import List, Tuple


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
		NOSE_INDEX, BASE_NECK_INDEX, CENTER_SPINE_INDEX,
		BASE_TAIL_INDEX, MID_TAIL_INDEX, TIP_TAIL_INDEX,
	],
]


def rle(inarray: np.ndarray):
	"""Run length encoding, implemented using numpy.

	Args:
		inarray: 1d vector

	Returns:
		tuple of (starts, durations, values)
		starts: start index of run
		durations: duration of run
		values: value of run
	"""
	ia = np.asarray(inarray)
	n = len(ia)
	if n == 0:
		return (None, None, None)
	else:
		y = ia[1:] != ia[:-1]
		i = np.append(np.where(y), n - 1)
		z = np.diff(np.append(-1, i))
		p = np.cumsum(np.append(0, z))[:-1]
		return (p, z, ia[i])


def argmax_2d(arr):
	"""Obtains the peaks for all keypoints in a pose for a single pose.

	Args:
		arr: np.ndarray of shape [1, 12, img_width, img_height]

	Returns:
		tuple of (values, coordinates)
		values: array of shape [12] containing the maximal values per-keypoint
		coordinates: array of shape [12, 2] containing the coordinates
	"""
	flatten_shape = list(arr.shape[:-2]) + [arr.shape[-1] * arr.shape[-2]]

	frame_idxs, _, max_rows, max_cols = np.unravel_index(np.argmax(arr.reshape(flatten_shape), axis=-1), arr.shape)
	keypoint_idxs = np.repeat([range(12)], repeats=len(frame_idxs), axis=-1)
	max_vals = arr[frame_idxs, keypoint_idxs, max_rows, max_cols]

	return max_vals, np.stack([max_rows, max_cols], -1).squeeze(0)


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
	for i, (start, duration) in enumerate(zip(rle_starts[rle_values == 1], rle_durations[rle_values == 1])):
		instance_track_id[start:start + duration] = i
	return pose_data_v3, conf_data_v3, instance_count, instance_embedding, instance_track_id


def render_pose_overlay(image: np.ndarray, frame_points: np.ndarray, exclude_points: List = [], color: Tuple = (255, 255, 255)) -> np.ndarray:
	"""Renders a single pose on an image.

	Args:
		image: image to render pose on
		frame_points: keypoints to render. keypoints are ordered [y, x]
		exclude_points: set of keypoint indices to exclude
		color: color to render the pose

	Returns:
		modified image
	"""
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
			[(pt_x, pt_y) for pt_y, pt_x in frame_points[curr_line_indexes]],
			np.int32)
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
