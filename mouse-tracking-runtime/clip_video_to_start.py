#!/usr/bin/env python3
"""Script to produce a clip of pose and video data based on when a mouse is first detected."""

import argparse
import subprocess
from pathlib import Path

import numpy as np
from utils import find_first_pose_file, write_pose_clip

SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60

def print_time(frames: int, fps: int = 30.0):
	"""Prints human readable frame times.

	Args:
		frames: number of frames to be translated
		fps: number of frames per second

	Returns:
		string representation of frames in H:M:S.s
	"""
	seconds = frames / fps
	if seconds < SECONDS_PER_MINUTE:
		return f'{np.round(seconds, 4)}s'
	minutes, seconds = divmod(seconds, SECONDS_PER_MINUTE)
	if minutes < MINUTES_PER_HOUR:
		return f'{minutes}m{np.round(seconds, 4)}s'
	hours, minutes = divmod(minutes, MINUTES_PER_HOUR)
	return f'{hours}h{minutes}m{np.round(seconds, 4)}s'


def clip_video(in_video, in_pose, out_video, out_pose, frame_start, frame_end):
	"""Clips a video and pose file.

	Args:
		in_video: path indicating the video to copy frames from
		in_pose: path indicating the pose file to copy frames from
		out_video: path indicating the output video
		out_pose: path indicating the output pose file
		frame_start: first frame in the video to copy
		frame_end: last frame in the video to copy

	Notes:
		This function requires ffmpeg to be installed on the system.
	"""
	if not Path(in_video).exists():
		msg = f'{in_video} does not exist'
		raise FileNotFoundError(msg)
	if not Path(in_pose).exists():
		msg = f'{in_pose} does not exist'
		raise FileNotFoundError(msg)
	if not isinstance(frame_start, int):
		msg = f'frame_start must be an integer, not {type(frame_start)}'
		raise TypeError(msg)
	if not isinstance(frame_end, int):
		msg = f'frame_start must be an integer, not {type(frame_end)}'
		raise TypeError(msg)

	ffmpeg_command = ['ffmpeg', '-hide_banner', '-loglevel', 'panic', '-r', '30', '-i', in_video, '-an', '-sn', '-dn', '-vf', f'select=gte(n\,{frame_start}),setpts=PTS-STARTPTS', '-vframes', f'{frame_end - frame_start}', '-f', 'mp4', '-c:v', 'libx264', '-preset', 'veryslow', '-profile:v', 'main', '-pix_fmt', 'yuv420p', '-g', '30', '-y', out_video]

	subprocess.run(ffmpeg_command, check=False)

	write_pose_clip(in_pose, out_pose, range(frame_start, frame_end))


def main():
	"""Command line interaction."""
	parser = argparse.ArgumentParser(description='Produce a video and pose clip aligned to criteria.')
	parser.add_argument('--in-video', help='input video file', required=True)
	parser.add_argument('--in-pose', help='input HDF5 pose file', required=True)
	parser.add_argument('--out-video', help='output video file', required=True)
	parser.add_argument('--out-pose', help='output HDF5 pose file', required=True)
	parser.add_argument('--allow-overwrite', help='Allows existing files to be overwritten (default error)', default=False, action='store_true')
	# Settings related to auto-detection
	parser.add_argument('--frame-offset', help='Number of frames to offset from the first detected pose. Positive values indicate adding time before. (Default 150)', type=int, default=150)
	parser.add_argument('--num-keypoints', help='Number of keypoints to consider a detected pose. (Default 12)', type=int, default=12)
	parser.add_argument('--confidence-threshold', help='Minimum confidence of a keypoint to be considered valid. (Default 0.3)', type=float, default=0.3)
	# Settings for clipping
	parser.add_argument('--observation-duration', help='Duration of the observation to clip. (Default 1hr)', type=int, default=30 * 60 * 60)

	args = parser.parse_args()
	if not args.allow_overwrite:
		if Path(args.out_video).exists():
			msg = f'{args.out_video} exists. If you wish to overwrite, please include --allow-overwrite'
			raise FileExistsError(msg)
		if Path(args.out_pose).exists():
			msg = f'{args.out_pose} exists. If you wish to overwrite, please include --allow-overwrite'
			raise FileExistsError(msg)

	first_frame = find_first_pose_file(args.in_pose, args.confidence_threshold, args.num_keypoints)
	output_start_frame = first_frame - args.frame_offset
	output_end_frame = output_start_frame + args.frame_offset + args.observation_duration
	print(f'Clipping video from frames {output_start_frame} ({print_time(output_start_frame)}) to {output_end_frame} ({print_time(output_end_frame)})')
	clip_video(args.in_video, args.in_pose, args.out_video, args.out_pose, output_start_frame, output_end_frame)


if __name__ == '__main__':
	main()
