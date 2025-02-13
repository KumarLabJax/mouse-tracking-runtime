# In branch of sleap_io that corrects static object xy sorting
# As of the time of writing this (2023-12-22), only Brian has access to this branch locally.
# However, these fixes should make their way into the official sleap_io repo.
# Check for a merged pull request related to this.
from sleap_io.io import jabs
from sleap_io.io import slp
import os
import re
import h5py
from pathlib import Path
import warnings
import numpy as np
from scipy.spatial.distance import cdist
corrected_file = '/media/bgeuther/Storage/TempStorage/dataset-releases/autism/cropped_videos/labels.v001.slp'
root_folder = '/media/bgeuther/Storage/TempStorage/dataset-releases/autism/cropped_videos/'
corrected_annotations = slp.read_labels(corrected_file)

def measure_pair_dists(annotation):
	dists = cdist(annotation, annotation)
	dists = dists[np.nonzero(np.triu(dists))]
	return dists


for video in corrected_annotations.videos:
	out_filename = str(Path(root_folder) / os.path.splitext(video.filename[0])[0]) + f"_pose_est_v6.h5"
	# Patch the 'frames' subfolder and replacing '/' with '+'
	out_filename = re.sub('frames/', '', re.sub('\+', '/', out_filename))
	if not os.path.exists(out_filename):
		warnings.warn(f"{out_filename} doesn't exist. Skipping...")
		continue
	data = jabs.convert_labels(corrected_annotations, video)
	jabs.write_static_objects(data, out_filename)
	# We also need to fix px_to_cm field
	# This code is adopted from tf-obj-api corner scripts
	coordinates = data['static_objects']['corners']
	dists = measure_pair_dists(coordinates)
	# Edges are shorter than diagonals
	sorted_dists = np.sort(dists)
	edges = sorted_dists[:4]
	diags = sorted_dists[4:]
	# Calculate all equivalent edge lengths (turn diagonals into edges)
	edges = np.concatenate([np.sqrt(np.square(diags) / 2), edges])
	cm_per_pixel = np.float32(52. / np.mean(edges))
	with h5py.File(out_filename, 'a') as f:
		f['poseest'].attrs['cm_per_pixel'] = cm_per_pixel
		f['poseest'].attrs['cm_per_pixel_source'] = 'manually_set'
