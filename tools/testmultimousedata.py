import argparse
import colorsys
import itertools
import matplotlib.pyplot as plt
import numpy as np
import random
import skimage.draw as skidraw

import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config

from dataset.multimousepose import MultiPoseDataset, parse_poses
import models


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
    parser = argparse.ArgumentParser(description='test the multimouse pose dataset')

    parser.add_argument('--cvat-files',
                        help='list of CVAT XML files to use',
                        nargs='+',
                        required=True,
                        type=str)
    parser.add_argument('--image-dir',
                        help='directory containing images',
                        required=True,
                        type=str)
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    update_config(cfg, args)

    normalize = transforms.Normalize(
        mean=[0.485], std=[0.229]
    )

    all_poses = list(itertools.chain.from_iterable(parse_poses(f) for f in args.cvat_files))
    mpose_ds = MultiPoseDataset(
        cfg,
        args.image_dir,
        all_poses,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
    )

    colors = random_colors(10)

    for _ in range(100):
        i = random.randrange(len(mpose_ds))
        print('doing', i)
        item = mpose_ds[i]

        image = item['image'][0, ...].numpy()

        plt.imshow(image, cmap='gray')
        plt.show()

        plt.imshow(item['joint_heatmaps'].numpy().max(0))
        plt.show()

        pose_instances = item['pose_instances'][:item['instance_count'], ...]
        inst_image = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.float32)
        inst_image_counts = np.zeros([image.shape[0], image.shape[1]], dtype=np.uint8)
        for instance_index, pose_instance in enumerate(pose_instances):
            for xy_point in pose_instance:
                temp_inst_image = np.zeros([image.shape[0], image.shape[1], 3], dtype=np.float32)
                rr, cc = skidraw.circle(xy_point[1], xy_point[0], 10, inst_image.shape)
                skidraw.set_color(temp_inst_image, (rr, cc), colors[instance_index])
                inst_image_counts[rr, cc] += 1
                inst_image += temp_inst_image
        inst_image /= np.expand_dims(inst_image_counts, 2)

        plt.imshow(inst_image * np.expand_dims(item['joint_heatmaps'].numpy().max(0), 2))
        plt.show()


if __name__ == "__main__":
    main()
