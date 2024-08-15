"""Script to downgrade a multi-mouse pose file into multiple single mouse pose files."""

import argparse
import re
import os
import h5py
from utils import write_pose_v2_data, write_pixel_per_cm_attr, convert_multi_to_v2, InvalidPoseFileException


def downgrade_pose_file(pose_h5_path, disable_id: bool = False):
	"""Downgrades a multi-mouse pose file into multiple single mouse pose files.

	Args:
		pose_h5_path: input pose file
		disable_id: bool to disable identity embedding tracks (if available) and use tracklet data instead
	"""
	if not os.path.isfile(pose_h5_path):
		raise FileNotFoundError(f'ERROR: missing file: {pose_h5_path}')
	# Read in all the necessary data
	with h5py.File(pose_h5_path, 'r') as pose_h5:
		if 'version' in pose_h5['poseest'].attrs:
			major_version = pose_h5['poseest'].attrs['version'][0]
		else:
			raise InvalidPoseFileException(f'Pose file {pose_h5_path} did not have a valid version.')
		if major_version == 2:
			print(f'Pose file {pose_h5_path} is already v2. Exiting.')
			exit(0)

		all_points = pose_h5['poseest/points'][:]
		all_confidence = pose_h5['poseest/confidence'][:]
		if major_version >= 4 and not disable_id:
			all_track_id = pose_h5['poseest/instance_embed_id'][:]
		elif major_version >= 3:
			all_track_id = pose_h5['poseest/instance_track_id'][:]
		try:
			config_str = pose_h5['poseest/points'].attrs['config']
			model_str = pose_h5['poseest/points'].attrs['model']
		except (KeyError, AttributeError):
			config_str = 'unknown'
			model_str = 'unknown'
		pose_attrs = pose_h5['poseest'].attrs
		if 'cm_per_pixel' in pose_attrs and 'cm_per_pixel_source' in pose_attrs:
			pixel_scaling = True
			px_per_cm = pose_h5['poseest'].attrs['cm_per_pixel']
			source = pose_h5['poseest'].attrs['cm_per_pixel_source']
		else:
			pixel_scaling = False

	downgraded_pose_data = convert_multi_to_v2(all_points, all_confidence, all_track_id)
	new_file_base = re.sub('_pose_est_v[0-9]+\\.h5', '', pose_h5_path)
	for animal_id, pose_data, conf_data in downgraded_pose_data:
		out_fname = f'{new_file_base}_animal_{animal_id}_pose_est_v2.h5'
		write_pose_v2_data(out_fname, pose_data, conf_data, config_str, model_str)
		if pixel_scaling:
			write_pixel_per_cm_attr(out_fname, px_per_cm, source)


def main():
	"""Command line interaction."""
	parser = argparse.ArgumentParser()
	parser.add_argument('--in-pose', help='input HDF5 pose file', required=True)
	parser.add_argument('--disable-id', help='forces tracklet ids (v3) to be exported instead of longterm ids (v4)', default=False, action='store_true')
	args = parser.parse_args()
	downgrade_pose_file(args.in_pose, args.disable_id)


if __name__ == '__main__':
	main()
