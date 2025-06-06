"""Script for aggregating fecal boli counts into a csv file."""

import numpy as np
import pandas as pd
import h5py
import glob
from datetime import datetime
import argparse
import sys


def aggregate_folder_data(folder: str, depth: int = 2, num_bins: int = -1):
	"""Aggregates fecal boli data in a folder into a table.

	Args:
		folder: project folder
		depth: expected subfolder depth
		num_bins: number of bins to read in (value < 0 reads all)

	Returns:
		pd.DataFrame containing the fecal boli counts over time

	Notes:
		Open field project folder looks like [computer]/[date]/[video]_pose_est_v6.h5 files
		depth defaults to have these 2 folders

	Todo:
		Currently this makes some bad assumptions about data.
			Time is assumed to be 1-minute intervals. Another field stores the times when they occur
			_pose_est_v6 is searched, but this is currently a proposed v7 feature
			no error handling is present...
	"""
	pose_files = glob.glob(folder + '/' + '*/' * depth + '*_pose_est_v6.h5')

	max_bin_count = None if num_bins < 0 else num_bins

	read_data = []
	for cur_file in pose_files:
		with h5py.File(cur_file, 'r') as f:
			counts = f['dynamic_objects/fecal_boli/counts'][:].flatten().astype(float)
			# Clip the number of bins if requested
			if max_bin_count is not None:
				if len(counts) > max_bin_count:
					counts = counts[:max_bin_count]
				elif len(counts) < max_bin_count:
					counts = np.pad(counts, (0, max_bin_count - len(counts)), 'constant', constant_values=np.nan)
		new_df = pd.DataFrame(counts, columns=['count'])
		new_df['minute'] = np.arange(len(new_df))
		new_df['NetworkFilename'] = cur_file[len(folder):len(cur_file) - 15] + '.avi'
		pivot = new_df.pivot(index='NetworkFilename', columns='minute', values='count')
		read_data.append(pivot)

	all_data = pd.concat(read_data).reset_index(drop=False)
	return all_data


def main(argv):
	"""Parse command line args and write out data."""
	parser = argparse.ArgumentParser(description='Script that generates a basic table of fecal boli counts for a project directory.')
	parser.add_argument('--folder', help='Folder containing the fecal boli prediction data', required=True)
	parser.add_argument('--folder_depth', help='Depth of the folder to search', type=int, default=2)
	parser.add_argument('--num_bins', help='Number of fecal boli bins to read in (default all)', type=int, default=-1)
	parser.add_argument('--output', help='Output table filename', default=f'FecalBoliCounts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

	args = parser.parse_args()
	df = aggregate_folder_data(args.folder, args.folder_depth, args.num_bins)
	df.to_csv(args.output, index=False, na_rep='NA')


if __name__ == '__main__':
	main(sys.argv[1:])
