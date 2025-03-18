#!/usr/bin/env python
"""Integrates SLEAP annotations of arena corners back into pose files."""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.distance import cdist
from sleap_io.io import jabs, slp

# Keys of static objects that are stored in y,x instead of x,y order
FLIPPED_OBJECTS = [
    "lixit",
    "food_hopper",
]

def measure_pair_dists(annotation):
    """Measure distances between all pairs of points.

    Args:
        annotation: Array of shape (n_points, 2)

    Returns:
        Distances between all pairs of points.
    """
    dists = cdist(annotation, annotation)
    return dists[np.nonzero(np.triu(dists))]


def write_static_objects(sleap_data: dict, filename: str):
    """Write static object data to a JABS pose file.

    Args:
        sleap_data: Dictionary of JABS data generated from convert_labels
        filename: Filename to write data to
    """
    with h5py.File(filename, "a") as h5:
        pose_grp = h5.require_group("poseest")
        pose_grp.attrs.update({"version": [5, 0]})
        if "static_objects" in sleap_data:
            object_grp = h5.require_group("static_objects")
            for object_key, object_keypoints in sleap_data["static_objects"].items():
                if object_key in FLIPPED_OBJECTS:
                    object_keypoints = np.flip(object_keypoints, axis=-1)
                if object_key in object_grp:
                    del object_grp[object_key]
                object_grp.require_dataset(
                    object_key,
                    object_keypoints.shape,
                    np.uint16,
                    data=object_keypoints.astype(np.uint16),
                )

def write_px_per_cm(sleap_data: dict, filename: str):
    """Write pixels per cm data to JABS pose file.

    Args:
        sleap_data: Dictionary of JABS data generated from convert_labels
        filename: Filename to write data to
    """
    coordinates = sleap_data['static_objects']['corners']
    dists = measure_pair_dists(coordinates)
    # Edges are shorter than diagonals
    sorted_dists = np.sort(dists)
    edges = sorted_dists[:4]
    diags = sorted_dists[4:]
    edges = np.concatenate([np.sqrt(np.square(diags) / 2), edges])
    cm_per_pixel = np.float32(52. / np.mean(edges))
    with h5py.File(filename, 'a') as f:
        f['poseest'].attrs['cm_per_pixel'] = cm_per_pixel
        f['poseest'].attrs['cm_per_pixel_source'] = 'manually_set'


def main(argv):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Script that integrates SLEAP annotations of arena corners back into pose files.')
    parser.add_argument('--sleap-annotations', help='SLEAP annotations file.', required=True)
    parser.add_argument('--pose-file', help='Pose file to correct.', required=True)

    args = parser.parse_args()
    if not Path(args.sleap_annotations).exists():
        msg = f"{args.sleap_annotations} doesn't exist."
        raise FileNotFoundError(msg)
    if not Path(args.pose_file).exists():
        msg = f"{args.pose_file} doesn't exist."
        raise FileNotFoundError(msg)

    # Load SLEAP annotations
    corrected_annotations = slp.read_labels(args.sleap_annotations)
    # Search for annotations for the requested pose file
    corrected_frame_names = [x.backend.filename[0] for x in  corrected_annotations.videos]
    expected_corrected_prefix = [Path(x).stem for x in corrected_frame_names]

    matched_video_idx = [i for i, x in enumerate(expected_corrected_prefix) if Path(args.pose_file).stem.startswith(x)]
    if len(matched_video_idx) == 0:
        msg = f"Couldn't find annotations for {args.pose_file}."
        raise ValueError(msg)

    video = corrected_annotations.videos[matched_video_idx[0]]
    data = jabs.convert_labels(corrected_annotations, video)
    # SLEAP uses 'arena_corners' key, while JABS just uses 'corners'
    data['static_objects']['corners'] = data['static_objects'].pop('arena_corners')
    write_static_objects(data, args.pose_file)
    write_px_per_cm(data, args.pose_file)

if __name__ == '__main__':
    main(sys.argv[1:])
