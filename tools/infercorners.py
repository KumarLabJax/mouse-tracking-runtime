import argparse
import h5py
import imageio
import numpy as np
import time
import yaml

import torch
import torch.nn.parallel
import torch.nn.functional as torchfunc
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config

import models
import cv2
import string
import os.path

import skimage

FRAMES_PER_MINUTE = 30 * 60


def argmax_2d(tensor):

    assert tensor.dim() >= 2
    max_col_vals, max_cols = torch.max(tensor, -1, keepdim=True)
    max_vals, max_rows = torch.max(max_col_vals, -2, keepdim=True)
    max_cols = torch.gather(max_cols, -2, max_rows)

    max_vals = max_vals.squeeze(-1).squeeze(-1)
    max_rows = max_rows.squeeze(-1).squeeze(-1)
    max_cols = max_cols.squeeze(-1).squeeze(-1)

    return max_vals, torch.stack([max_rows, max_cols], -1)

# Example use:
#
#   time python -u tools/infercorners.py \
#       --model-file output-full-mouse-pose/hdf5mousepose/pose_hrnet/corner-detection/model_best.pth \
#       --cfg corner-detection.yaml \
#       --root-dir ~/smb/labshare \
#       --batch-file netfiles.csv
#
#   time python -u tools/infercorners.py \
#       --model-file output-corner/simplepoint/pose_hrnet/corner_2020-06-30_01/best_state.pth \
#       --cfg experiments/corner/corner_2020-06-30_01.yaml \
#       --root-dir ~/smb/labshare \
#       --batch-file /home/sheppk/projects/massimo-deep-hres-net/netfiles.csv

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cfg',
        help='the configuration for the model to use for inference',
    )

    parser.add_argument(
        '--model-file',
        help='the model file to use for inference',
        default=None,
    )

    parser.add_argument(
        '--batch-file',
        help='the batch file listing videos to process',
    )

    parser.add_argument(
        '--root-dir',
        help='the root directory that batch file paths are build off of'
    )

    # parser.add_argument(
    #     'video',
    #     help='the input video',
    # )

    # parser.add_argument(
    #     'poseout',
    #     help='the pose estimation output HDF5 file',
    # )

    args = parser.parse_args()
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    if args.model_file:
        cfg.TEST.MODEL_FILE = args.model_file
    cfg.freeze()

    start_time = time.time()

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

    xform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.45, 0.45, 0.45],
            std=[0.225, 0.225, 0.225],
        ),
    ])

    # mockup = None

    with torch.no_grad(), open(args.batch_file) as batch_file:
        for line in batch_file:
            vid_filename = line.strip()
            if vid_filename:
                video_filename = os.path.join(args.root_dir, vid_filename)
                if os.path.isfile(video_filename):

                    with imageio.get_reader(video_filename) as reader:

                        all_preds = []
                        all_maxvals = []
                        batch = []

                        def perform_inference():
                            if batch:
                                batch_tensor = torch.stack([xform(img) for img in batch]).cuda()
                                batch.clear()

                                x = model(batch_tensor)

                                x.squeeze_(-3)

                                img_w = 480
                                img_h = 480

                                x_ul = x[:, :(img_w // 2), :(img_h // 2)]
                                x_ur = x[:, (img_w // 2):img_w, :(img_h // 2)]
                                x_ll = x[:, :(img_w // 2), (img_w // 2):img_h]
                                x_lr = x[:, (img_w // 2):img_w, (img_h // 2):img_h]

                                maxvals1, preds1 = argmax_2d(x_ul)
                                maxvals2, preds2 = argmax_2d(x_ur)
                                maxvals3, preds3 = argmax_2d(x_ll)
                                maxvals4, preds4 = argmax_2d(x_lr)

                                maxvals1 = maxvals1.cpu().numpy()
                                maxvals2 = maxvals2.cpu().numpy()
                                maxvals3 = maxvals3.cpu().numpy()
                                maxvals4 = maxvals4.cpu().numpy()

                                preds1 = preds1.cpu().numpy().astype(np.uint16)
                                preds2 = preds2.cpu().numpy().astype(np.uint16)
                                preds3 = preds3.cpu().numpy().astype(np.uint16)
                                preds4 = preds4.cpu().numpy().astype(np.uint16)

                                preds2[..., 0] += img_w // 2
                                preds3[..., 1] += img_h // 2
                                preds4[..., 0] += img_w // 2
                                preds4[..., 1] += img_h // 2

                                predStack = np.stack([preds1, preds2, preds3, preds4], axis=-2)
                                maxvalStack = np.stack([maxvals1, maxvals2, maxvals3, maxvals4], axis=-1)

                                all_preds.append(predStack)
                                all_maxvals.append(maxvalStack)

                        last_frame_index = 600
                        frame_step_size = 100
                        for frame_index, image in enumerate(reader):

                            if frame_index == 0:
                                mockup = image

                            if frame_index % frame_step_size == 0:

                                batch.append(image)
                                # print("frame: %d" % frame_index)
                                perform_inference()

                            if frame_index == last_frame_index:
                                break

                        all_preds = np.concatenate(all_preds)
                        all_maxvals = np.concatenate(all_maxvals)

                        xmed1 = []
                        xmed2 = []
                        xmed3 = []
                        xmed4 = []

                        ymed1 = []
                        ymed2 = []
                        ymed3 = []
                        ymed4 = []

                        for i in range(len(all_preds[0])):
                            xmed1.append(all_preds[i, 0, 0])
                            xmed2.append(all_preds[i, 1, 0])
                            xmed3.append(all_preds[i, 2, 0])
                            xmed4.append(all_preds[i, 3, 0])

                            ymed1.append(all_preds[i, 0, 1])
                            ymed2.append(all_preds[i, 1, 1])
                            ymed3.append(all_preds[i, 2, 1])
                            ymed4.append(all_preds[i, 3, 1])

                        xs = [
                            int(np.median(xmed1)),
                            int(np.median(xmed2)),
                            int(np.median(xmed3)),
                            int(np.median(xmed4)),
                        ]
                        ys = [
                            int(np.median(ymed1)),
                            int(np.median(ymed2)),
                            int(np.median(ymed3)),
                            int(np.median(ymed4)),
                        ]
                        out_doc = {
                            'corner_coords': {
                                'xs': xs,
                                'ys': ys,
                            }
                        }

                        video_filename_root, _ = os.path.splitext(video_filename)
                        video_yaml_out_filename = video_filename_root + '_corners_v2.yaml'
                        print('Writing to:', video_yaml_out_filename)
                        with open(video_yaml_out_filename, 'w') as video_yaml_out_file:
                            yaml.safe_dump(out_doc, video_yaml_out_file)

                        video_png_out_filename = video_filename_root + '_corners_v2.png'
                        for i in range(4):
                            rr, cc = skimage.draw.circle(ys[i], xs[i], 5, mockup.shape)
                            skimage.draw.set_color(mockup, (rr, cc), [255, 0, 0])
                        skimage.io.imsave(video_png_out_filename, mockup)


if __name__ == "__main__":
    main()
