import argparse
import itertools
import matplotlib.pyplot as plt
import random

import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config

from dataset.OpenFieldObjDataset import OpenFieldObjDataset, parse_obj_labels
import models

def main():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
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
    obj_labels = list(parse_obj_labels(cfg.DATASET.CVAT_XML))
    ofods = OpenFieldObjDataset(
        cfg,
        obj_labels,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
    )

    for _ in range(100):
        i = random.randrange(len(ofods))
        print('doing', i)
        img, target = ofods[i]

        plt.imshow(img[0, ...].numpy(), cmap='gray')
        plt.show()
        plt.imshow(target.numpy())
        plt.show()


if __name__ == "__main__":
    main()
