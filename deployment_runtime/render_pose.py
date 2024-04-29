"""Main script for rendering pose file related data onto a video."""

import argparse
import imageio
import os
import h5py
from utils import render_pose_overlay, render_segmentation_overlay, plot_keypoints, convert_v2_to_v3


static_obj_colors = {
	'lixit': (55, 126, 184),       # Water spout is Blue
	'food_hopper': (255, 127, 0),  # Food hopper is Orange
	'corners': (75, 175, 74),      # Arena corners are Green
}

# Are the static objects stored as [x, y] sorting?
static_obj_xy = {
	'lixit': False,
	'food_hopper': False,
	'corners': True,
}

# Taken from colorbrewer2 Qual Set1 and Qual Paired
# Some colors were removed due to overlap with static object colors
mouse_colors = [
	(228, 26, 28),    # Red
	(152, 78, 163),   # Purple
	(255, 255, 51),   # Yellow
	(166, 86, 40),    # Brown
	(247, 129, 191),  # Pink
	(166, 206, 227),  # Light Blue
	(178, 223, 138),  # Light Green
	(251, 154, 153),  # Peach
	(253, 191, 111),  # Light Orange
	(202, 178, 214),  # Light Purple
	(255, 255, 153),  # Faded Yellow
]


def process_video(in_video_path, pose_h5_path, out_video_path, disable_id: bool = False):
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
		raise FileNotFoundError(f'ERROR: missing file: {in_video_path}')
	if not os.path.isfile(pose_h5_path):
		raise FileNotFoundError(f'ERROR: missing file: {pose_h5_path}')
	# Read in all the necessary data
	with h5py.File(pose_h5_path, 'r') as pose_h5:
		if 'version' in pose_h5['poseest'].attrs:
			major_version = pose_h5['poseest'].attrs['version'][0]
		else:
			major_version = 2
		all_points = pose_h5['poseest/points'][:]
		# v6 stores segmentation data
		if major_version >= 6:
			all_seg_data = pose_h5['poseest/seg_data'][:]
			if not disable_id:
				all_seg_id = pose_h5['poseest/longterm_seg_id'][:]
			else:
				all_seg_id = pose_h5['poseest/instance_seg_id'][:]
		else:
			all_seg_data = None
			all_seg_id = None
		# v5 stores optional static object data.
		all_static_object_data = {}
		if major_version >= 5:
			for key in pose_h5['static_objects'].keys():
				all_static_object_data[key] = pose_h5[f'static_objects/{key}'][:]
		# v4 stores identity/tracklet merging data
		if major_version >= 4 and not disable_id:
			all_track_id = pose_h5['poseest/instance_embed_id'][:]
		elif major_version >= 3:
			all_track_id = pose_h5['poseest/instance_track_id'][:]
		# Data is v2, upgrade it to v3
		else:
			conf_data = pose_h5['poseest/confidence'][:]
			all_points, _, _, _, all_track_id = convert_v2_to_v3(all_points, conf_data)

	# Process the video
	with imageio.get_reader(in_video_path) as video_reader, imageio.get_writer(out_video_path, fps=30) as video_writer:
		for frame_index, image in enumerate(video_reader):
			for obj_key, obj_data in all_static_object_data.items():
				# Arena corners are TL, TR, BL, BR, so sort them into a correct polygon for plotting
				# TODO: possibly use `sort_corners`?
				if obj_key == 'corners':
					obj_data = obj_data[[0, 1, 3, 2]]
				image = plot_keypoints(obj_data, image, color=static_obj_colors[obj_key], is_yx=not static_obj_xy[obj_key], include_lines=obj_key != 'lixit')
			for pose_idx, pose_id in enumerate(all_track_id[frame_index]):
				image = render_pose_overlay(image, all_points[frame_index, pose_idx], color=mouse_colors[pose_id % len(mouse_colors)])
			if all_seg_data is not None:
				for seg_idx, seg_id in enumerate(all_seg_id[frame_index]):
					image = render_segmentation_overlay(all_seg_data[frame_index, seg_idx], image, color=mouse_colors[seg_id % len(mouse_colors)])
			video_writer.append_data(image)
	print(f'finished generating video: {out_video_path}', flush=True)


def main():
	"""Command line interaction."""
	parser = argparse.ArgumentParser()
	parser.add_argument('--in-vid', help='input video to process', required=True)
	parser.add_argument('--in-pose', help='input HDF5 pose file', required=True)
	parser.add_argument('--out-vid', help='output pose overlay video to generate', required=True)
	parser.add_argument('--disable-id', help='forces track ids (v3) to be plotted instead of embedded identity (v4)', default=False, action='store_true')
	args = parser.parse_args()
	process_video(args.in_vid, args.in_pose, args.out_vid, args.disable_id)


if __name__ == '__main__':
	main()
