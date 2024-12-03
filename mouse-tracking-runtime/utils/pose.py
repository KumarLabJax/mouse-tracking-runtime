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
	"""Obtains the peaks for all keypoints in a pose.

	Args:
		arr: np.ndarray of shape [batch, 12, img_width, img_height]

	Returns:
		tuple of (values, coordinates)
		values: array of shape [batch, 12] containing the maximal values per-keypoint
		coordinates: array of shape [batch, 12, 2] containing the coordinates
	"""
	full_max_cols = np.argmax(arr, axis=-1, keepdims=True)
	max_col_vals = np.take_along_axis(arr, full_max_cols, axis=-1)
	max_rows = np.argmax(max_col_vals, axis=-2, keepdims=True)
	max_row_vals = np.take_along_axis(max_col_vals, max_rows, axis=-2)
	max_cols = np.take_along_axis(full_max_cols, max_rows, axis=-2)

	max_vals = max_row_vals.squeeze(-1).squeeze(-1)
	max_idxs = np.stack([max_rows.squeeze(-1).squeeze(-1), max_cols.squeeze(-1).squeeze(-1)], axis=-1)

	return max_vals, max_idxs


def get_peak_coords(arr):
	"""Converts a boolean array of peaks into locations.

	Args:
		arr: array of shape [w, h] to search for peaks

	Returns:
		tuple of (values, coordinates)
		values: array of shape [n_peaks] containing the maximal values per-peak
		coordinates: array of shape [n_peaks, 2] containing the coordinates
	"""
	peak_locations = np.argwhere(arr)
	if len(peak_locations) == 0:
		return np.zeros([0], dtype=np.float32), np.zeros([0, 2], dtype=np.int16)

	max_vals = []
	for coord in peak_locations:
		max_vals.append(arr[coord.tolist()])

	return np.stack(max_vals), peak_locations


def localmax_2d(arr, threshold, radius):
	"""Obtains the multiple peaks with non-max suppression.

	Args:
		arr: np.ndarray of shape [img_width, img_height]
		threshold: threshold required for a positive to be found
		radius: square radius (rectangle, not circle) peaks must be apart to be considered a peak. Largest peaks will cause all other potential peaks in this radius to be omitted.

	Returns:
		tuple of (values, coordinates)
		values: array of shape [n_peaks] containing the maximal values per-peak
		coordinates: array of shape [n_peaks, 2] containing the coordinates
	"""
	assert radius >= 1
	assert np.squeeze(arr).ndim == 2

	point_heatmap = np.expand_dims(np.squeeze(arr), axis=-1)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius * 2 + 1, radius * 2 + 1))
	# Non-max suppression
	dilated = cv2.dilate(point_heatmap, kernel)
	mask = arr >= dilated
	eroded = cv2.erode(point_heatmap, kernel)
	mask_2 = arr > eroded
	mask = np.logical_and(mask, mask_2)
	# Peakfinding via Threshold
	mask = np.logical_and(mask, arr > threshold)
	bool_arr = np.full(dilated.shape, False, dtype=bool)
	bool_arr[mask] = True
	return get_peak_coords(bool_arr)


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
			duplicated_frames = sorted_frames[:-1][sorted_frames[1:] == sorted_frames[:-1]]
			raise ValueError(f'Identity {cur_id} contained multiple poses assigned on frames {duplicated_frames}.')
		single_pose = np.zeros([len(pose_data), 12, 2], dtype=pose_data.dtype)
		single_conf = np.zeros([len(pose_data), 12], dtype=conf_data.dtype)
		single_pose[id_frames] = pose_data[id_frames, id_idxs]
		single_conf[id_frames] = conf_data[id_frames, id_idxs]

		return_list.append((cur_id, single_pose, single_conf))

	return return_list


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
