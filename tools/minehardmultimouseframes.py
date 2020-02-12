import argparse
import imageio
import itertools
import os

import torch
import torch.backends.cudnn as cudnn

import _init_paths
import utils.assocembedutil as aeutil
from infermultimousepose import infer_pose_instances
from config import cfg
from config import update_config

import models

# Example:
#
#   share_root='/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar'
#   python -u tools/minehardmultimouseframes.py \
#       --videos "${share_root}"/NV1-B2B/2019-10-2[23]/*.avi \
#       --root-dir "${share_root}" \
#       --outdir hard_frames \
#       --max-embed-sep-within-instances 0.3 \
#       --min-embed-sep-between-instances 0.3 \
#       ./output-multi-mouse/multimousepose/pose_hrnet/multimouse_2020-01-17_11/best_state.pth \
#       ./experiments/multimouse/multimouse_2020-01-17_11.yaml
#
#   share_root='/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar'
#   python -u tools/minehardmultimouseframes.py \
#       --videos \
#           "${share_root}"/NV5-CBAX2/2019-11-22/MDX0089_2019-11-2*_19-00-00.avi \
#           "${share_root}"/NV5-CBAX2/2019-11-22/MDX0089_2019-11-2*_00-00-00.avi \
#           "${share_root}"/NV12-B2B/2019-10-31/*.avi \
#           "${share_root}"/NV16-UCSD/2019-10-14/3879439_2019-10-14_19-21-39.avi \
#           "${share_root}"/NV16-UCSD/2019-10-14/3879439_2019-10-15_10-00-00.avi \
#           "${share_root}"/NV16-UCSD/2019-10-14/3879439_2019-10-15_04-00-00.avi \
#           "${share_root}"/NV7-CBAX2/2019-11-22/MDX0090_2019-11-*_19-00-00.avi \
#           "${share_root}"/NV7-CBAX2/2019-11-22/MDX0090_2019-11-*_00-00-00.avi \
#           "${share_root}"/NV13-B2B/2019-10-24/*.avi \
#           "${share_root}"/NV5-CBAX2/2019-11-14/MDX0077_2019-11-1*_19-00-00.avi \
#           "${share_root}"/NV5-CBAX2/2019-11-14/MDX0077_2019-11-1*_00-00-00.avi \
#       --root-dir "${share_root}" \
#       --outdir hard_frames \
#       --max-embed-sep-within-instances 0.3 \
#       --min-embed-sep-between-instances 0.2 \
#       --min-pose-heatmap-val 1.5 \
#       ./output-multi-mouse/multimousepose/pose_hrnet/multimouse_2020-02-03_06/best_state.pth \
#       ./experiments/multimouse/multimouse_2020-02-03_06.yaml

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model_file',
        help='the model file to use for inference',
    )
    parser.add_argument(
        'cfg',
        help='the configuration for the model to use for inference',
    )
    # TODO we should change this to cm units rather than pixels
    parser.add_argument(
        '--max-inst-dist-px',
        help='maximum keypoint separation distance in pixels. For a keypoint to '
             'be added to an instance there must be at least one point in the '
             'instance which is within this number of pixels.',
        type=int,
        default=150,
    )
    parser.add_argument(
        '--max-embed-sep-within-instances',
        help='maximum embedding separation allowed for a joint to be added to an existing '
             'instance within the max distance separation',
        type=float,
        default=0.2,
    )
    parser.add_argument(
        '--min-embed-sep-between-instances',
        help='if two joints of the the same type (eg. both right ear) are within the max '
             'distance separation and their embedding separation doesn\'t meet or '
             'exceed this threshold only the point with the highest heatmap value is kept.',
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '--min-pose-heatmap-val',
        type=float,
        default=0.4,
    )
    parser.add_argument(
        '--max-pose-dist-px',
        type=float,
        default=40,
    )
    parser.add_argument(
        '--min-joint-count',
        help='if a pose instance has fewer than this number of points it is discarded',
        type=int,
        default=6,
    )
    parser.add_argument(
        '--max-instance-count',
        help='a frame should not contain more than this number of poses. If it does, extra poses '
             'will be discarded in order of least confidence until we meet this threshold.',
        type=int,
    )
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
        '--outdir',
        required=True,
        help='the output directory',
    )
    parser.add_argument(
        '--frames-per-vid',
        type=int,
        default=10,
        help='how many frames to output per video',
    )

    args = parser.parse_args()

    root_dir = os.path.normpath(args.root_dir)

    # shorten some args
    max_embed_sep = args.max_embed_sep_within_instances
    min_embed_sep = args.min_embed_sep_between_instances
    max_inst_dist = args.max_inst_dist_px

    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.TEST.MODEL_FILE = args.model_file
    cfg.freeze()

    # start_time = time.time()

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )
    print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    model.eval()
    model = model.cuda()

    model_extra = cfg.MODEL.EXTRA
    use_neighboring_frames = False
    if 'USE_NEIGHBORING_FRAMES' in model_extra:
        use_neighboring_frames = model_extra['USE_NEIGHBORING_FRAMES']

    root_dir = os.path.normpath(args.root_dir)

    os.makedirs(args.outdir, exist_ok=True)
    for vid_fname in args.videos:
        print('Processing:', vid_fname)
        net_id = os.path.relpath(os.path.normpath(vid_fname), root_dir)

        with imageio.get_reader(vid_fname) as frame_reader:
            frame_reader1, frame_reader2 = itertools.tee(frame_reader, 2)
            pose_instances = infer_pose_instances(
                    model, frame_reader1,
                    use_neighboring_frames,
                    min_embed_sep, max_embed_sep, max_inst_dist,
                    args.min_joint_count, args.max_instance_count, args.max_pose_dist_px,
                    args.min_pose_heatmap_val)

            # TODO hardcoding these for now but they should probably be converted to command line args
            MIN_FRAMES_BETWEEN_SAVES = 30 * 10
            MIN_TRACK_POSE_DISTANCE = 100
            MIN_TRACK_FRAMES = 30 * 2

            track_dicts = dict()
            last_hard_frame_index = -MIN_FRAMES_BETWEEN_SAVES
            hard_frame_count = 0
            save_next_frame = False
            prev_frame = None
            frame_pose_zip = zip(itertools.count(), frame_reader2, pose_instances)
            for frame_index, frame, frame_pose_instances in frame_pose_zip:
                frame_is_hard = False
                save_curr_frame = False
                save_prev_frame = False

                enough_hard_frames = hard_frame_count >= args.frames_per_vid
                enough_frames_between_saves = (
                    frame_index - last_hard_frame_index >= MIN_FRAMES_BETWEEN_SAVES)

                if save_next_frame:
                    save_curr_frame = True
                    save_next_frame = False

                frame_pose_dict = {
                    pose.instance_track_id : pose
                    for pose in frame_pose_instances}
                for track_id, track_dict in list(track_dicts.items()):
                    if track_id not in frame_pose_dict:

                        if not frame_is_hard and not enough_hard_frames and enough_frames_between_saves:
                            # A track has dissapeared. If the first and last pose have enough
                            # movement (distance) we will call this a hard frame.
                            track_pose_distance = aeutil.pose_distance(
                                track_dict['first_pose'],
                                track_dict['last_pose'])
                            track_frames = frame_index - track_dict['start_frame_index']
                            if track_pose_distance >= MIN_TRACK_POSE_DISTANCE and track_frames >= MIN_TRACK_FRAMES:
                                frame_is_hard = True

                        del track_dicts[track_id]

                if frame_is_hard:
                    save_prev_frame = True
                    save_curr_frame = True
                    save_next_frame = True
                    last_hard_frame_index = frame_index
                    hard_frame_count += 1

                if save_prev_frame and prev_frame is not None:
                    frame_fname = '{}_{}.png'.format(
                        net_id.replace('/', '+').replace('\\', '+'),
                        frame_index - 1)
                    print('SAVING:', frame_fname)
                    imageio.imwrite(os.path.join(args.outdir, frame_fname), prev_frame)

                if save_curr_frame:
                    frame_fname = '{}_{}.png'.format(
                        net_id.replace('/', '+').replace('\\', '+'),
                        frame_index)
                    print('SAVING:', frame_fname)
                    imageio.imwrite(os.path.join(args.outdir, frame_fname), frame)

                for track_id, pose in frame_pose_dict.items():
                    if track_id in track_dicts:
                        track_dicts[track_id]['last_pose'] = pose
                    else:
                        track_dicts[track_id] = {
                            'start_frame_index': frame_index,
                            'first_pose': pose,
                            'last_pose': pose,
                        }

                prev_frame = frame
                if enough_hard_frames and not save_next_frame:
                    break


if __name__ == "__main__":
    main()
