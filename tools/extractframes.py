import argparse
import imageio
import itertools
import math
import numpy as np
import os


# Example:
#
#   share_root='/home/sheppk/smb/labshare'
#   python tools/extractframes.py \
#       --root-dir "${share_root}" \
#       --videos \
#           "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-22_23-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-23_11-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-23_23-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-24_12-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-25_00-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-25_11-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-26_06-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-26_21-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-27_11-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-28_01-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-28_16-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-29_09-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-29_23-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-04-30_11-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-05-01_06-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-05-02_02-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-05-02_18-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-05-03_07-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-05-03_21-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-22/MDX0159_2020-05-04_11-00-00.avi \
#           "${share_root}"/NV5-CBAX2/2020-03-26/MDX0148_2020-03-26_17-00-00.avi "${share_root}"/NV5-CBAX2/2020-03-26/MDX0148_2020-03-27_06-00-00.avi "${share_root}"/NV5-CBAX2/2020-03-26/MDX0148_2020-03-28_00-00-00.avi "${share_root}"/NV5-CBAX2/2020-03-26/MDX0148_2020-03-28_16-00-00.avi "${share_root}"/NV5-CBAX2/2020-03-26/MDX0148_2020-03-29_06-00-00.avi "${share_root}"/NV5-CBAX2/2020-03-26/MDX0148_2020-03-29_21-00-00.avi \
#       --frame-indexes 600 \
#       --outdir fecal-boli-image-batch4

#   python tools/extractframes.py \
#       --root-dir "${share_root}" \
#       --videos \
#           "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-09_22-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-10_09-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-11_01-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-11_13-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-12_01-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-12_14-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-13_01-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-13_13-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-14_00-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-14_12-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-14_23-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-15_13-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-16_01-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-16_13-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-17_03-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-17_17-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-18_11-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-19_03-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-19_14-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-20_07-00-00.avi "${share_root}"/NV5-CBAX2/2020-04-09/MDX0159_2020-04-21_00-00-00.avi \
#       --frame-indexes 600 \
#       --outdir fecal-boli-image-batch4


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--videos',
        nargs='+',
        help='the input videos',
    )
    parser.add_argument(
        '--root-dir',
        required=True,
        help='when determining video network ID this prefix root is stripped from the video name',
    )
    parser.add_argument(
        '--frame-indexes',
        type=int,
        nargs='+',
        help='the frame indexes to extract',
    )
    parser.add_argument(
        '--outdir',
        required=True,
        help='the output directory',
    )

    args = parser.parse_args()

    root_dir = os.path.normpath(args.root_dir)

    for vid_fname in args.videos:
        print('Processing:', vid_fname)
        net_id = os.path.relpath(os.path.normpath(vid_fname), root_dir)

        frame_indexes = sorted(args.frame_indexes)
        os.makedirs(args.outdir, exist_ok=True)
        with imageio.get_reader(vid_fname) as reader:
            for frame_index in frame_indexes:
                img_data = reader.get_data(frame_index)
                frame_fname = '{}_{}.png'.format(
                    net_id.replace('/', '+').replace('\\', '+'),
                    frame_index)
                imageio.imwrite(os.path.join(args.outdir, frame_fname), img_data)


if __name__ == "__main__":
    main()
