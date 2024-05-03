"""Inference script for mouse identity model."""

import argparse
import sys
import os
from tfs_inference import infer_multi_identity_tfs


def main(argv):
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(description='Script that infers a mouse identity model.')
	parser.add_argument('--model', help='Trained model to infer.', default='social-paper', choices=['social-paper'])
	parser.add_argument('--runtime', help='Runtime to execute the model.', default='tfs', choices=['tfs'])
	vid_or_img = parser.add_mutually_exclusive_group(required=True)
	vid_or_img.add_argument('--video', help='Video file for processing')
	vid_or_img.add_argument('--frame', help='Image file for processing')
	parser.add_argument('--out-file', help='Pose file to write out.', required=True)
	#
	args = parser.parse_args()
	if args.video:
		assert os.path.exists(args.video)
	else:
		assert os.path.exists(args.frame)
	if args.runtime == 'tfs':
		infer_multi_identity_tfs(args)


if __name__ == '__main__':
	main(sys.argv[1:])
