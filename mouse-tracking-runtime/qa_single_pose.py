#!/usr/bin/env python3
"""Script for aggregating fecal boli counts into a csv file."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from utils import inspect_pose_v6


def main(argv):
	"""Parse command line args and write out data."""
	parser = argparse.ArgumentParser(description='Script that generates a tabular quality metrics for a single mouse pose file.')
	parser.add_argument('--pose', help='Pose file to inspect.', required=True)
	parser.add_argument('--output', help='Output filename. Will append row if already exists.', default=f'QA_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
	parser.add_argument('--pad', help='Number of frames to pad the start and end of the video.', type=int, default=150)
	parser.add_argument('--duration', help='Duration of the video in frames.', type=int, default=108000)

	args = parser.parse_args()
	quality_df = pd.DataFrame(inspect_pose_v6(args.pose, args.pad, args.duration), index=[0])
	quality_df.to_csv(args.output, mode='a', index=False, header=not Path(args.output).exists())


if __name__ == '__main__':
	main(sys.argv[1:])
