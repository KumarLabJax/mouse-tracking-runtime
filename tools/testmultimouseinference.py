import argparse
import colorsys
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import skimage
import torch
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import _init_paths
import utils.assocembedutil as aeutil
from config import cfg
from config import update_config

from dataset.multimousepose import MultiPoseDataset, parse_poses
import models


def xy_dist(pt1, pt2):
    x_diff = pt2['x_pos'] - pt1['x_pos']
    y_diff = pt2['y_pos'] - pt1['y_pos']

    return math.sqrt(x_diff ** 2 + y_diff ** 2)


class PoseInstance(object):

    def __init__(self):
        self.keypoints = dict()

        self._sum_x_pos = 0
        self._sum_y_pos = 0
        self._sum_inst_embed = 0

    @property
    def mean_inst_embed(self):
        return self._sum_inst_embed / len(self.keypoints)

    def add_keypoint(self, keypoint):

        assert keypoint['joint_index'] not in self.keypoints
        self.keypoints[keypoint['joint_index']] = keypoint

        self._sum_inst_embed += keypoint['embed']

    def nearest_dist(self, keypoint):
        min_dist = None
        for pose_keypoint in self.keypoints.values():
            curr_dist = xy_dist(keypoint, pose_keypoint)

            if min_dist is None or curr_dist < min_dist:
                min_dist = curr_dist

        return min_dist


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = [colorsys.hsv_to_rgb(*c) for c in hsv]
    #random.shuffle(colors)
    return colors


def main():

    colors = random_colors(6)

    parser = argparse.ArgumentParser(description='test the multimouse pose dataset')

    parser.add_argument(
        '--cvat-files',
        help='list of CVAT XML files to use',
        nargs='+',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--image-dir',
        help='directory containing images',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--image-list',
        help='file containing newline separated list of images to use',
        default=None,
    )
    parser.add_argument(
        '--model-file',
        help='the model file to use for inference',
        required=True,
    )
    parser.add_argument(
        '--confidence-threshold',
        help='minimum confidence threshold to test',
        default=0.0,
        type=float,
    )
    parser.add_argument(
        '--cfg',
        help='the configuration for the model to use for inference',
        required=True,
        type=str,
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
        '--image-out-dir',
        help='the directory we plot to',
    )

    args = parser.parse_args()

    # shorten some args
    min_embed_sep = args.min_embed_sep_between_instances
    max_inst_dist = args.max_inst_dist_px

    if args.image_out_dir is not None:
        os.makedirs(args.image_out_dir, exist_ok=True)

    print('=> loading configuration from {}'.format(args.cfg))

    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    if args.model_file:
        cfg.TEST.MODEL_FILE = args.model_file
    cfg.freeze()

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

    normalize = transforms.Normalize(
        mean=[0.485], std=[0.229]
    )

    image_list_filename = args.image_list
    img_names = None
    if image_list_filename is not None:
        img_names = set()
        with open(image_list_filename) as val_file:
            for curr_line in val_file:
                img_name = curr_line.strip()
                img_names.add(img_name)

    pose_labels = list(itertools.chain.from_iterable(parse_poses(f) for f in args.cvat_files))
    if img_names is not None:
        pose_labels = [p for p in pose_labels if p['image_name'] in img_names]

    for pose_label in pose_labels:
        image_name = pose_label['image_name']
        # pose_instances = pose_label['pose_instances']

        image_path = os.path.join(args.image_dir, image_name)
        image_data_numpy = skimage.io.imread(image_path, as_gray=True)

        print('image_name:', image_name)

        image_data = torch.from_numpy(image_data_numpy).to(torch.float32)
        image_data = normalize(image_data.unsqueeze(0)).squeeze(0)
        image_data = torch.stack([image_data] * 3).unsqueeze(0)
        image_data = image_data.cuda()

        inst_pose_data = model(image_data)
        joint_count = inst_pose_data.size(1) // 2
        pose_heatmaps = inst_pose_data[:, :joint_count, ...]
        inst_embed_data = inst_pose_data[:, joint_count:, ...]

        pose_localmax = aeutil.localmax2D(pose_heatmaps, 0.4, 3)

        for batch_index in range(1):
            pose_instances = []
            for joint_index in range(joint_count):
                joint_localmax = pose_localmax[batch_index, joint_index, ...]

                joint_xy = joint_localmax.nonzero().cpu()
                joint_xy[...] = joint_xy[..., [1, 0]].clone()

                joint_embed = inst_embed_data[batch_index, joint_index, ...]
                joint_embed = joint_embed[joint_localmax].cpu()

                pose_heatmap = pose_heatmaps[batch_index, joint_index, ...]
                pose_conf = pose_heatmap[joint_localmax].cpu()

                joint_insts = []
                for inst_index in range(joint_xy.size(0)):
                    joint_insts.append({
                        'joint_index': joint_index,
                        'x_pos': joint_xy[inst_index, 0].item(),
                        'y_pos': joint_xy[inst_index, 1].item(),
                        'conf': pose_conf[inst_index].item(),
                        'embed': joint_embed[inst_index].item(),
                    })

                # Here we remove any keypoints that are both spatially too close and too
                # close in the embedding space. In these cases the joint with higher confidence
                # is kept and the other is discarded
                joint_insts.sort(key=lambda j: j['conf'])
                joint_insts_filtered = []
                for inst_index1, joint_inst1 in enumerate(joint_insts):
                    min_embed_sep_violated = False
                    for joint_inst2 in joint_insts[inst_index1 + 1:]:
                        if (abs(joint_inst1['embed'] - joint_inst2['embed']) < min_embed_sep
                                and xy_dist(joint_inst1, joint_inst2) <= max_inst_dist):
                            min_embed_sep_violated = True
                            break

                    if not min_embed_sep_violated:
                        joint_insts_filtered.append(joint_inst1)
                joint_insts_filtered.reverse()
                joint_insts = joint_insts_filtered

                # TODO pick one of these two methods and delete the other
                if True:
                    for joint_inst in joint_insts:
                        best_pose_match = None
                        best_embed_diff = None

                        # find nearest instance in embedding space
                        for pose_instance in pose_instances:
                            if joint_index not in pose_instance.keypoints:
                                embed_diff = abs(joint_inst['embed'] - pose_instance.mean_inst_embed)
                                if best_embed_diff is None or embed_diff < best_embed_diff:
                                    spatial_dist = pose_instance.nearest_dist(joint_inst)
                                    if spatial_dist <= max_inst_dist:
                                        best_pose_match = pose_instance
                                        best_embed_diff = embed_diff

                        if best_pose_match is None:
                            # since there's no existing pose match create a new one
                            best_pose_match = PoseInstance()
                            pose_instances.append(best_pose_match)

                        best_pose_match.add_keypoint(joint_inst)
                else:
                    for pose_instance in pose_instances:
                        best_keypoint_index = None
                        best_embed_diff = None

                        for keypoint_index, joint_inst in enumerate(joint_insts):
                            embed_diff = abs(joint_inst['embed'] - pose_instance.mean_inst_embed)
                            if best_embed_diff is None or embed_diff < best_embed_diff:
                                spatial_dist = pose_instance.nearest_dist(joint_inst)
                                if spatial_dist <= max_inst_dist:
                                    best_keypoint_index = keypoint_index
                                    best_embed_diff = embed_diff

                        if best_keypoint_index is not None:
                            best_keypoint = joint_insts[best_keypoint_index]
                            del joint_insts[best_keypoint_index]
                            pose_instance.add_keypoint(best_keypoint)

                    for joint_inst in joint_insts:
                        pose_instance = PoseInstance()
                        pose_instance.add_keypoint(joint_inst)
                        pose_instances.append(pose_instance)

            image_rgb = np.zeros([image_data_numpy.shape[0], image_data_numpy.shape[1], 3], dtype=np.float32)
            image_rgb[...] = image_data_numpy[..., np.newaxis]

            print('==== POSES ====')
            for pose_index, pose_instance in enumerate(pose_instances):
                print(
                    'Pose {} ({} points):'.format(pose_index, len(pose_instance.keypoints)),
                    sorted(pose_instance.keypoints.values(), key=lambda kp: kp['joint_index']),
                )
                print()

                for keypoint in pose_instance.keypoints.values():
                    rr, cc = skimage.draw.circle(
                        keypoint['y_pos'], keypoint['x_pos'],
                        3,
                        image_rgb.shape)
                    skimage.draw.set_color(image_rgb, (rr, cc), colors[pose_index % len(colors)])

            if args.image_out_dir is not None:
                _, axs = plt.subplots(1, 2, figsize=(12,6))
                axs[0].imshow(image_rgb, aspect='equal')

                for pose_index, pose_instance in enumerate(pose_instances):
                    keypoints = sorted(pose_instance.keypoints.values(), key=lambda kp: kp['joint_index'])
                    joint_indexes = [kp['joint_index'] for kp in keypoints]
                    embed_vals = [kp['embed'] for kp in keypoints]
                    axs[1].scatter(embed_vals, joint_indexes, c=[colors[pose_index % len(colors)]])

                image_base, _ = os.path.splitext(os.path.basename(image_name))
                plt.savefig(os.path.join(
                    args.image_out_dir,
                    image_base + '_instance_pose.png'))


if __name__ == "__main__":
    main()
