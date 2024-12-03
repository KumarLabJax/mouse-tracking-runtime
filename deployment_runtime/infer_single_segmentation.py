"""Inference script for single mouse pose model."""

import argparse
import sys
import os
from tfs_inference import infer_single_segmentation_tfs


def main(argv):
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(description='Script that infers an onnx single mouse segmentation model.')
	parser.add_argument('--model', help='Trained model to infer.', default='tracking-paper', choices=['tracking-paper'])
	parser.add_argument('--runtime', help='Runtime to execute the model.', default='tfs', choices=['tfs'])
	vid_or_img = parser.add_mutually_exclusive_group(required=True)
	vid_or_img.add_argument('--video', help='Video file for processing')
	vid_or_img.add_argument('--frame', help='Image file for processing')
	parser.add_argument('--out-file', help='Pose file to write out.', required=True)
	parser.add_argument('--out-video', help='Render the results to a video.', default=None)
	#
	args = parser.parse_args()
	if args.video:
		assert os.path.exists(args.video)
	else:
		assert os.path.exists(args.frame)
	if args.runtime == 'tfs':
		infer_single_segmentation_tfs(args)


if __name__ == '__main__':
	main(sys.argv[1:])
