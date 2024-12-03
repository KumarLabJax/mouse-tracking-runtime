"""Script to patch [y, x] to [x, y] sorting of static object data."""

import h5py
import numpy as np
import argparse


def swap_static_obj_xy(pose_file, object_key):
	"""Swaps the [y, x] data to [x, y] for a given static object key.

	Args:
		pose_file: pose file to modify in-place
		object_key: dataset key to swap x and y data
	"""
	with h5py.File(pose_file, 'a') as f:
		if object_key not in f:
			print(f'{object_key} not in {pose_file}.')
			return
		object_data = np.flip(f[object_key][:], axis=-1)
		if len(f[object_key].attrs.keys()) > 0:
			object_attrs = dict(f[object_key].attrs.items())
		else:
			object_attrs = {}
		compression_opt = f[object_key].compression_opts

		del f[object_key]

		if compression_opt is None:
			f.create_dataset(object_key, data=object_data)
		else:
			f.create_dataset(object_key, data=object_data, compression='gzip', compression_opts=compression_opt)
		for cur_attr, data in object_attrs.items():
			f[object_key].attrs.create(cur_attr, data)


def main():
	"""Command line interaction."""
	parser = argparse.ArgumentParser()
	parser.add_argument('--in-pose', help='input HDF5 pose file', required=True)
	parser.add_argument('--object-key', help='data key to swap the sorting of [y, x] data to [x, y]', required=True)
	args = parser.parse_args()
	swap_static_obj_xy(args.in_pose, args.object_key)


if __name__ == '__main__':
	main()
