import argparse
import itertools
import matplotlib.pyplot as plt
import random

import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config

import dataset
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

    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )
    normalize = transforms.Normalize(
        mean=[0.485], std=[0.229]
    )
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    for _ in range(100):
        i = random.randrange(len(train_dataset))
        print('doing', i)
        input, target, target_weight, meta = train_dataset[i]

        plt.imshow(input[0, ...].numpy(), cmap='gray')
        plt.show()
        plt.imshow(target.sum(dim=0).numpy())
        plt.show()


if __name__ == "__main__":
    main()
