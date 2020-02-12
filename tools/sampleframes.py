import argparse
import imageio
import itertools
import math
import numpy as np
import os


# Example:
#
#   share_root='/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar'
#   python tools/sampleframes.py \
#       --videos "${share_root}"/NV1-B2B/2019-10-2[23]/*.avi \
#       --root-dir "${share_root}" \
#       --outdir sampled_frames \
#       --include-neighbor-frames
#
#   share_root='/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar'
#   python tools/sampleframes.py \
#       --videos "${share_root}"/NV1-B2B/2019-10-2[23]/*.avi \
#       --root-dir "${share_root}" \
#       --outdir sampled_frames \
#       --include-neighbor-frames
#
#   share_root='/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar'
#   python tools/sampleframes.py \
#       --videos \
#           "${share_root}"/NV16-UCSD/2019-10-09/3879434_2019-10-09_20-00-00.avi \
#           "${share_root}"/NV16-UCSD/2019-10-11/3879436_2019-10-12_13-00-00.avi \
#           "${share_root}"/NV16-UCSD/2019-10-14/3879439_2019-10-15_03-00-00.avi \
#       --root-dir "${share_root}" \
#       --outdir sampled_frames_UCSD \
#       --include-neighbor-frames



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
        '--frames-per-vid',
        type=int,
        default=10,
        help='how many frames to output per video',
    )
    parser.add_argument(
        '--outdir',
        required=True,
        help='the output directory',
    )
    parser.add_argument(
        '--include-neighbor-frames',
        action='store_true',
        help='extract neighboring frames too (ie for frame n we also save frame n-1 and n+1)'
    )

    args = parser.parse_args()

    root_dir = os.path.normpath(args.root_dir)

    for vid_fname in args.videos:
        print('Processing:', vid_fname)
        net_id = os.path.relpath(os.path.normpath(vid_fname), root_dir)

        video_len = 0
        with imageio.get_reader(vid_fname) as reader:
            video_len = reader.get_length()
            if not math.isfinite(video_len):
                video_len = 0
                for _ in reader:
                    video_len += 1

        assert video_len >= 30 * 60, vid_fname + ' is less than a minute long'

        frames_to_sample = np.random.choice(video_len, args.frames_per_vid, replace=False)

        if args.include_neighbor_frames:
            frames_to_sample = sorted(set(itertools.chain.from_iterable(
                (max(f - 1, 0), f, min(f + 1, video_len - 1))
                for f in frames_to_sample)))
        else:
            frames_to_sample = sorted(frames_to_sample)

        os.makedirs(args.outdir, exist_ok=True)
        with imageio.get_reader(vid_fname) as reader:
            for frame_index in frames_to_sample:
                img_data = reader.get_data(frame_index)
                frame_fname = '{}_{}.png'.format(
                    net_id.replace('/', '+').replace('\\', '+'),
                    frame_index)
                imageio.imwrite(os.path.join(args.outdir, frame_fname), img_data)


if __name__ == "__main__":
    main()
