"""Inference script for lixit spout."""

import argparse
import sys
import os
from lightning_inference import infer_fecal_boli_lightning as infer_lightning


def main(argv):
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(description='Script that infers a lixit water spout model.')
	parser.add_argument('--model', help='Trained model to infer.', default='fecal-boli', choices=['fecal-boli'])
	parser.add_argument('--runtime', help='Runtime to execute the model.', default='lightning', choices=['lightning'])
	vid_or_img = parser.add_mutually_exclusive_group(required=True)
	vid_or_img.add_argument('--video', help='Video file for processing')
	vid_or_img.add_argument('--frame', help='Image file for processing')
	parser.add_argument('--out-file', help='Pose file to write out.', default=None)
	parser.add_argument('--out-image', help='Render the final prediction to an image.', default=None)
	parser.add_argument('--out-video', help='Render all predictions to a video.', default=None)
	parser.add_argument('--frame-interval', help='Interval of frames to predict on.', default=1800, type=int)
	parser.add_argument('--batch-size', help='atch size to use while making predictions.', default=1, type=int)
	#
	args = parser.parse_args()
	if args.video:
		assert os.path.exists(args.video)
	else:
		assert os.path.exists(args.frame)
	if args.runtime == 'lightning':
		infer_lightning(args)


if __name__ == '__main__':
	main(sys.argv[1:])
