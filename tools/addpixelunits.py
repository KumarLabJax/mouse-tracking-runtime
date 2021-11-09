import argparse
import h5py
import numpy as np
import os
from pathlib import Path, WindowsPath
import yaml

CORNERS_SUFFIX = '_corners_v2.yaml'
ARENA_SIZE_CM = 52

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--arena-size-cm',
        type=float,
        default=52,
        help='the arena size is used to derive cm/pixel using corners files',
    )
    parser.add_argument(
        'pose-file',
        help='Pose file embed pixel measurements into'
    )
    parser.add_argument(
        'corner-file',
        help='Corner file used to add pixel measurements'
    )

    args = parser.parse_args()

    with open(corners_path) as args.corner_file:
        corners_dict = yaml.safe_load(corners_file)
        #print(list(corners_dict.keys()))
        xs = corners_dict['corner_coords']['xs']
        ys = corners_dict['corner_coords']['ys']

        # get all of the non-diagonal pixel distances between
        # corners and take the meadian
        xy_ul, xy_ll, xy_ur, xy_lr = [
            np.array(xy, dtype=np.float) for xy in zip(xs, ys)
        ]
        med_corner_dist_px = np.median([
            np.linalg.norm(xy_ul - xy_ll),
            np.linalg.norm(xy_ll - xy_lr),
            np.linalg.norm(xy_lr - xy_ur),
            np.linalg.norm(xy_ur - xy_ul),
        ])

        cm_per_pixel = np.float32(args.arena_size_cm / med_corner_dist_px)
        with h5py.File(pose_path, 'r+') as args.pose_file:
            pose_h5_file['poseest'].attrs['cm_per_pixel'] = cm_per_pixel
            pose_h5_file['poseest'].attrs['cm_per_pixel_source'] = 'corner_detection'


if __name__ == '__main__':
    main()
