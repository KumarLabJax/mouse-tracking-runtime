"""Inference script for single mouse pose model."""

import argparse
import sys
import os
from pytorch_inference import infer_multi_pose_pytorch


def main(argv):
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(description='Script that infers an onnx single mouse pose model.')
	parser.add_argument('--model', help='Trained model to infer.', default='social-paper-topdown', choices=['social-paper-topdown'])
	parser.add_argument('--runtime', help='Runtime to execute the model.', default='pytorch', choices=['pytorch'])
	vid_or_img = parser.add_mutually_exclusive_group(required=True)
	vid_or_img.add_argument('--video', help='Video file for processing')
	vid_or_img.add_argument('--frame', help='Image file for processing')
	parser.add_argument('--out-file', help='Pose file to write out.', required=True)
	parser.add_argument('--out-video', help='Render the results to a video.', default=None)
	parser.add_argument('--batch-size', help='Batch size to use while making predictions.', default=1, type=int)
	#
	args = parser.parse_args()
	if args.video:
		assert os.path.exists(args.video)
	else:
		assert os.path.exists(args.frame)
	if not os.path.exists(args.out_file):
		raise ValueError(f'Pose file containing segmentation data is required. Pose file {args.out_file} is empty.')
	if args.runtime == 'pytorch':
		infer_multi_pose_pytorch(args)


if __name__ == '__main__':
	main(sys.argv[1:])
