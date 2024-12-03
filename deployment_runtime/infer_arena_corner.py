"""Inference script for arena corners."""

import argparse
import sys
import os
from tfs_inference import infer_arena_corner_model as infer_tfs


def main(argv):
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(description='Script that infers an onnx single mouse pose model.')
	parser.add_argument('--model', help='Trained model to infer.', default='social-2022-pipeline', choices=['social-2022-pipeline'])
	parser.add_argument('--runtime', help='Runtime to execute the model.', default='tfs', choices=['tfs'])
	vid_or_img = parser.add_mutually_exclusive_group(required=True)
	vid_or_img.add_argument('--video', help='Video file for processing')
	vid_or_img.add_argument('--frame', help='Image file for processing')
	parser.add_argument('--out-file', help='Pose file to write out.', default=None)
	parser.add_argument('--out-image', help='Render the final prediction to an image.', default=None)
	parser.add_argument('--out-video', help='Render all predictions to a video.', default=None)
	parser.add_argument('--num-frames', help='Number of frames to predict on.', default=100, type=int)
	parser.add_argument('--frame-interval', help='Interval of frames to predict on.', default=100, type=int)
	#
	args = parser.parse_args()
	if args.video:
		assert os.path.exists(args.video)
	else:
		assert os.path.exists(args.frame)
	if args.runtime == 'tfs':
		infer_tfs(args)


if __name__ == '__main__':
	main(sys.argv[1:])
