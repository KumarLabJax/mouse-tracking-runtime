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
#
#   python tools/sampleframes.py \
#       --videos \
#             ../gaitanalysis/spot-check/LL1-1_002105-M-AX12-5.28571428571429-42640-1-S331.avi \
#             ../gaitanalysis/spot-check/LL1-3_000690-M-MP13-8-42416-3-S080.avi \
#             ../gaitanalysis/spot-check/LL1-3_001800-M-MP16-10-42409-3-S099.avi \
#             ../gaitanalysis/spot-check/LL1-4_002105-F-AX12-5.28571428571429-42640-4-S329.avi \
#             ../gaitanalysis/spot-check/LL2-2_002019-M-AX30-10.2857142857143-42864-3-S420.avi \
#             ../gaitanalysis/spot-check/LL2-3_000674-F-AX18-5-42726-1-S393.avi \
#             ../gaitanalysis/spot-check/LL2-4_002105-M-AX12-5.28571428571429-42640-8-S332.avi \
#             ../gaitanalysis/spot-check/LL2-4_LP.avi \
#             ../gaitanalysis/spot-check/LL3-1_000687-M-AX11-7.71428571428571-42630-1-S320.avi \
#             ../gaitanalysis/spot-check/LL3-2_000674-F-AX18-10-42691-4-S393.avi \
#             ../gaitanalysis/spot-check/LL3-2_000687-M-AX11-6.42857142857143-42639-2-S337.avi \
#             ../gaitanalysis/spot-check/LL3-2_002019-F-AX30-8.28571428571429-42878-2-S421.avi \
#             ../gaitanalysis/spot-check/LL4-1_005314-F-AX5-9-42423-3-S137.avi \
#             ../gaitanalysis/spot-check/LL4-3_000674-M-AX18-4.71428571428571-42728-1-S395.avi \
#             ../gaitanalysis/spot-check/LL4-3_000690-F-MP13-8-42402-5-S018.avi \
#             ../gaitanalysis/spot-check/LL4-4_000676-M-AX29-10.2857142857143-42864-3-S422.avi \
#             ../gaitanalysis/spot-check/LL5-3_000928-M-AX1-8-42423-5-S123.avi \
#             ../gaitanalysis/spot-check/LL5-4_001800-F-MP16-8-42409-1-S026.avi \
#             ../gaitanalysis/spot-check/LL5-4_CAST_F.avi \
#             ../gaitanalysis/spot-check/LL6-1_000674-M-AX18-5-42726-7-S396.avi \
#             ../gaitanalysis/spot-check/LL6-1_000687-F-AX11-6.42857142857143-42639-4-S323.avi \
#             ../gaitanalysis/spot-check/LL6-2_TALLYHOJngJ.avi \
#             ../gaitanalysis/spot-check/LL6-3_000676-M-AX29-8-42409-7-S091.avi \
#             ../gaitanalysis/spot-check/LL6-3_FVB_F.avi \
#             ../gaitanalysis/spot-check/LL6-4_000687-F-AX11-7.71428571428571-42630-1-S323.avi \
#       --root-dir "../gaitanalysis/spot-check" \
#       --outdir fecal-boli-image-batch4 \
#       --frames-per-vid 1



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
