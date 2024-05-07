"""Script to stitch tracklets within a pose file."""

import h5py
import numpy as np
import argparse
from utils.matching import VideoObservations
from utils.writers import write_pose_v3_data, write_pose_v4_data, write_v6_tracklets


def match_predictions(pose_file):
	"""Reads in pose and segmentation data to match data over the time dimension.

	Args:
		pose_file: pose file to modify in-place

	Notes:
		This function only applies the optimal settings from identity repository.
	"""
	video_observations = VideoObservations.from_pose_file(pose_file, 0.0)
	video_observations.generate_greedy_tracklets(rotate_pose=True, num_threads=2)
	with h5py.File(pose_file, 'r') as f:
		pose_shape = f['poseest/points'].shape[:2]
		seg_shape = f['poseest/seg_data'].shape[:2]
	new_pose_ids, new_seg_ids = video_observations.get_id_mat(pose_shape, seg_shape)

	# Stitch the tracklets together
	video_observations.stitch_greedy_tracklets(num_tracks=None, prioritize_long=True)
	translated_tracks = video_observations.stitch_translation
	stitched_pose = np.vectorize(lambda x: translated_tracks.get(x, 0))(new_pose_ids)
	stitched_seg = np.vectorize(lambda x: translated_tracks.get(x, 0))(new_seg_ids)
	centers = video_observations.get_embed_centers()
	# Write data out

	# We need to overwrite original tracklet data
	write_pose_v3_data(pose_file, instance_track=new_pose_ids)
	# Also overwrite stitched tracklet data
	mask = new_pose_ids == 0
	write_pose_v4_data(pose_file, mask, stitched_pose, centers)
	# Finally, overwrite segmentation data
	write_v6_tracklets(pose_file, new_seg_ids, stitched_seg)


def main():
	"""Command line interaction."""
	parser = argparse.ArgumentParser()
	parser.add_argument('--in-pose', help='input HDF5 pose file', required=True)
	args = parser.parse_args()
	match_predictions(args.in_pose)


if __name__ == '__main__':
	main()
