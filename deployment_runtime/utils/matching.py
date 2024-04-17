"""Functions related to matching poses with segmentation."""
import numpy as np
import cv2
import scipy
from .segmentation import get_contour_stack
from typing import List


def get_point_dist(contour: List[np.ndarray], point: np.ndarray):
	"""Return the signed distance between a point and a contour.

	Args:
		contour: list of opencv-compliant contours
		point: point of shape [2]

	Returns:
		The largest value "inside" any contour in the list of contours

	Note:
		OpenCV point polygon test defines the signed distance as inside (positive), outside (negative), and on the contour (0).
		Here, we return negative as "inside".
	"""
	best_dist = -9999
	for contour_part in contour:
		cur_dist = cv2.pointPolygonTest(contour_part, tuple(point), measureDist=True)
		if cur_dist > best_dist:
			best_dist = cur_dist
	return -best_dist


def compare_pose_and_contours(contours: np.ndarray, poses: np.ndarray):
	"""Returns a masked 3D array of signed distances between the pose points and contours.

	Args:
		contours: matrix contour data of shape [n_animals, n_contours, n_points, 2]
		poses: pose data of shape [n_animals, n_keypoints, 2]

	Returns:
		distance matrix between poses and contours of shape [n_valid_poses, n_valid_contours, n_points]

	Notes:
		The shapes are not necessarily the same as the input matrices based on detected default values.
	"""
	num_poses = np.sum(~np.all(np.all(poses == 0, axis=2), axis=1))
	num_points = np.shape(poses)[1]
	contour_lists = [get_contour_stack(contours[x]) for x in np.arange(np.shape(contours)[0])]
	num_segs = np.count_nonzero(np.array([len(x) for x in contour_lists]))
	if num_poses == 0 or num_segs == 0:
		return None
	dists = np.ma.array(np.zeros([num_poses, num_segs, num_points]), mask=False)
	# TODO: Change this to a vectorized op
	for cur_point in np.arange(num_points):
		for cur_pose in np.arange(num_poses):
			for cur_seg in np.arange(num_segs):
				if np.all(poses[cur_pose, cur_point] == 0):
					dists.mask[cur_pose, cur_seg, cur_point] = True
				else:
					dists[cur_pose, cur_seg, cur_point] = get_point_dist(contour_lists[cur_seg], tuple(poses[cur_pose, cur_point]))
	return dists


def make_pose_seg_dist_mat(points: np.ndarray, seg_contours: np.ndarray, ignore_tail: bool = True, use_expected_dists: bool = False):
	"""Helper function to compare poses with contour data.

	Args:
		points: keypoint data for mice of shape [n_animals, n_points, 2] sorted (y, x)
		seg_contours: contour data of shape [n_animals, n_contours, n_points, 2] sorted (x, y)
		ignore_tail: bool to exclude 2 tail keypoints (11 and 12)
		use_expected_dists: adjust distances relative to where the keypoint should be on the mouse

	Returns:
		distance matrix from `compare_pose_and_contours`

	Note: This is a convenience function to run `compare_pose_and_contours` and adjust it more abstractly.
	"""
	# Flip the points
	# Also remove the tail points if requested
	if ignore_tail:
		# Remove points 11 and 12, which are mid-tail and tail-tip
		points_mat = np.copy(np.flip(points[:, :11, :], axis=-1))
	else:
		points_mat = np.copy(np.flip(points, axis=-1))
	dists = compare_pose_and_contours(seg_contours, points_mat)
	# Early return if no comparisons were made
	if dists is None:
		return np.ma.array(np.zeros([0, 2], dtype=np.uint32))
	# Suggest matchings based on results
	if not use_expected_dists:
		dists = np.mean(dists, axis=2)
	else:
		# Values of "20" are about midline of an average mouse
		expected_distances = np.array([0, 0, 0, 20, 0, 0, 20, 0, 0, 0, 0, 0])
		# Subtract expected distance
		dists = np.mean(dists - expected_distances[:np.shape(points_mat)[1]], axis=2)
		# Shift to describe "was close to expected"
		dists = -np.abs(dists) + 5
	dists.fill_value = -1
	return dists


def hungarian_match_points_seg(points: np.ndarray, seg_contours: np.ndarray, ignore_tail: bool = True, use_expected_dists: bool = False, max_dist: float = 0):
	"""Applies a hungarian matching algorithm to link segs and poses.

	Args:
		points: keypoint data of shape [n_animals, n_points, 2] sorted (y, x)
		seg_contours: padded contour data of shape [n_animals, n_contours, n_points, 2] sorted x, y
		ignore_tail: bool to exclude 2 tail keypoints (11 and 12)
		use_expected_dists: adjust distances relative to where the keypoint should be on the mouse
		max_dist: maximum distance to allow a match. Value of 0 means "average keypoint must be within the segmentation"

	Returns:
		matchings between pose and segmentations of shape [match_idx, 2] where each row is a match between [pose, seg] indices
	"""
	dists = make_pose_seg_dist_mat(points, seg_contours, ignore_tail, use_expected_dists)
	# TODO:
	# Add in filtering out non-unique matches
	hungarian_matches = np.asarray(scipy.optimize.linear_sum_assignment(dists)).T
	filtered_matches = np.array(np.zeros([0, 2], dtype=np.uint32))
	for potential_match in hungarian_matches:
		if dists[potential_match[0], potential_match[1]] < max_dist:
			filtered_matches = np.append(filtered_matches, [potential_match], axis=0)
	return filtered_matches
