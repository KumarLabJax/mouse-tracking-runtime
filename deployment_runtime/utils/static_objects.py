import numpy as np
import cv2
from typing import Tuple
from scipy.spatial.distance import cdist

ARENA_SIZE_CM = 20.5 * 2.54  # 20.5 inches to cm

DEFAULT_CM_PER_PX = {
	'ltm': ARENA_SIZE_CM / 701,  # 700.570 +/- 10.952 pixels
	'ofa': ARENA_SIZE_CM / 398,  # 397.992 +/- 8.069 pixels
}

ARENA_IMAGING_RESOLUTION = {
	800: 'ltm',
	480: 'ofa',
}


def plot_keypoints(kp: np.ndarray, img: np.ndarray, color: Tuple = (0, 0, 255)) -> np.ndarray:
	"""Plots keypoints on an image.

	Args:
		kp: keypoints of shape [n_keypoints, 2]
		img: image to render the keypoint on
		color: BGR tuple to render the keypoint

	Returns:
		Copy of image with the keypoints rendered
	"""
	img_copy = img.copy()
	for i, kp_data in enumerate(kp):
		_ = cv2.circle(img_copy, (int(kp_data[1]), int(kp_data[0])), 3, color, 2)
	return img_copy


def measure_pair_dists(keypoints: np.ndarray):
	"""Measures pairwise distances between all keypoints.

	Args:
		keypoints: keypoints of shape [n_points, 2]

	Returns:
		Distances of shape [n_comparisons]
	"""
	dists = cdist(keypoints, keypoints)
	dists = dists[np.nonzero(np.triu(dists))]
	return dists


def filter_square_keypoints(predictions: np.ndarray, tolerance: float = 25.0):
	"""Filters raw predictions for a square object.

	Args:
		predictions: raw predictions of shape [n_predictions, 4, 2]
		tolerance: allowed pixel variation

	Returns:
		Proposed actual keypoint locations of shape [4, 2]

	Raises:
		ValueError if predictions fail the tolerance test
	"""
	assert len(predictions.shape) == 3

	filtered_predictions = []
	for i in np.arange(len(predictions)):
		dists = measure_pair_dists(predictions[i])
		sorted_dists = np.sort(dists)
		edges, diags = np.split(sorted_dists, [4], axis=0)
		compare_edges = np.concatenate([np.sqrt(np.square(diags)/2), edges])
		edge_err = np.abs(compare_edges-np.mean(compare_edges))
		if np.all(edge_err < tolerance):
			filtered_predictions.append(predictions[i])

	if len(filtered_predictions) == 0:
		raise ValueError('No predictions were square.')
	filtered_predictions = np.stack(filtered_predictions)

	keypoint_motion = np.std(filtered_predictions, axis=0)
	keypoint_motion = np.hypot(keypoint_motion[:, 0], keypoint_motion[:, 1])

	if np.any(keypoint_motion > tolerance):
		raise ValueError('Good predictions are moving!')

	return np.mean(filtered_predictions, axis=0)


def get_px_per_cm(corners: np.ndarray, arena_size_cm: float = ARENA_SIZE_CM) -> float:
	"""Calculates the pixels per cm conversion for corner predictions.

	Args:
		corners: corner prediction data of shape [4, 2]

	Returns:
		coefficient to multiply pixels to get cm
	"""
	dists = measure_pair_dists(corners)
	# Edges are shorter than diagonals
	sorted_dists = np.sort(dists)
	edges = sorted_dists[:4]
	diags = sorted_dists[4:]
	# Calculate all equivalent edge lengths (turn diagonals into equivalent edges)
	edges = np.concatenate([np.sqrt(np.square(diags) / 2), edges])
	cm_per_pixel = np.float32(arena_size_cm / np.mean(edges))

	return cm_per_pixel
