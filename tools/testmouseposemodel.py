import argparse
import h5py
import numpy as np
import time

import matplotlib.pyplot as plt

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
from core.inference import get_final_preds
from core.inference import get_max_preds

import dataset
import models


NOSE_INDEX = 0

LEFT_EAR_INDEX = 1
RIGHT_EAR_INDEX = 2

BASE_NECK_INDEX = 3

LEFT_FRONT_PAW_INDEX = 4
RIGHT_FRONT_PAW_INDEX = 5

CENTER_SPINE_INDEX = 6

LEFT_REAR_PAW_INDEX = 7
RIGHT_REAR_PAW_INDEX = 8

BASE_TAIL_INDEX = 9
MID_TAIL_INDEX = 10
TIP_TAIL_INDEX = 11

INDEX_NAMES = [
    'Nose',

    'Left Ear',
    'Right Ear',

    'Base Neck',

    'Left Front Paw',
    'Right Front Paw',

    'Center Spine',

    'Left Rear Paw',
    'Right Rear Paw',

    'Base Tail',
    'Mid Tail',
    'Tip Tail',
]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-file',
        help='the model file to use for inference',
        default=None,
    )

    parser.add_argument(
        '--show-err-thresh',
        help='for each keypoint where pixel error is >= this threshold we print a message and'
             ' show a debug image.',
        default=None,
        type=float,
    )

    # parser.add_argument(
    #     '--out-dir',
    #     help='the output directory to use',
    #     default='temp',
    # )

    parser.add_argument(
        'cfg',
        help='the configuration for the model to use for inference',
    )

    args = parser.parse_args()
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    if args.model_file:
        cfg.TEST.MODEL_FILE = args.model_file
    cfg.freeze()

    print('=> loading configuration from {}'.format(args.cfg))

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
            mean=[0.45],
            std=[0.225],
        ),
    ])

    with torch.no_grad():

        with h5py.File(cfg.DATASET.ROOT, 'r') as hdf5file:
            l2_pixel_err_sum = None
            l2_pixel_err_max = None
            frame_count = 0
            for name, group in hdf5file[cfg.DATASET.TEST_SET].items():
                if 'frames' in group and 'points' in group:
                    points = group['points']
                    for grp_frame_index in range(points.shape[0]):
                        grp_frame_pts = points[grp_frame_index, ...]

                        data_numpy = group['frames'][grp_frame_index, ...]
                        data_numpy = data_numpy.squeeze(2)
                        data = xform(data_numpy).squeeze(0)
                        data = data.cuda()
                        data = torch.stack([data] * 3)
                        data = data.unsqueeze(0)

                        # print(grp_frame_pts.shape)
                        # print(data.shape)

                        inf_out = model(data)
                        in_out_ratio = data.size(-1) // inf_out.size(-1)
                        if in_out_ratio == 4:
                            # print('need to upscale')
                            inf_out = torchfunc.upsample(inf_out, scale_factor=4, mode='bicubic', align_corners=False)
                        inf_out = inf_out.cpu().numpy()
                        # print('inf_out.shape:', inf_out.shape)

                        preds, maxvals = get_max_preds(inf_out)
                        preds = preds.astype(np.uint16)

                        # print(preds)

                        # print('diff:', preds.dtype, grp_frame_pts.dtype)
                        pixel_err = preds.astype(np.float32) - grp_frame_pts
                        # print(pixel_err)

                        l2_pixel_err = np.linalg.norm(pixel_err, ord=2, axis=2)
                        # print(l2_pixel_err)

                        if args.show_err_thresh is not None:
                            over_thresh = np.transpose(np.nonzero(
                                l2_pixel_err >= args.show_err_thresh))

                            for curr_over_thresh in over_thresh:
                                print('Bad value for:   ', INDEX_NAMES[curr_over_thresh[1]])
                                print('Confidence:      ', maxvals[curr_over_thresh[0], curr_over_thresh[1]])
                                print('Pixel Distance:  ', l2_pixel_err[curr_over_thresh[0], curr_over_thresh[1]])

                                curr_heatmap = inf_out[curr_over_thresh[0], curr_over_thresh[1], ...]
                                curr_heatmap = np.ma.masked_where(curr_heatmap < 0.1, curr_heatmap)
                                plt.imshow(data_numpy, cmap='gray', vmin=0, vmax=255)
                                plt.imshow(curr_heatmap, cmap=plt.get_cmap('YlOrRd'), alpha=0.4)
                                plt.show()

                        if l2_pixel_err_sum is None:
                            l2_pixel_err_sum = l2_pixel_err.copy()
                            l2_pixel_err_max = l2_pixel_err.copy()
                        else:
                            l2_pixel_err_sum += l2_pixel_err
                            l2_pixel_err_max = np.maximum(l2_pixel_err_max, l2_pixel_err)

                        frame_count += 1

        l2_pixel_err_mean = l2_pixel_err_sum / frame_count
        l2_pixel_err_mean = l2_pixel_err_mean.squeeze(0)
        l2_pixel_err_max = l2_pixel_err_max.squeeze(0)

        print('L2 Pixel Error Mean of Means:  ', l2_pixel_err_mean.mean(), l2_pixel_err_max.max())
        print('NOSE Pixel Error:              ', l2_pixel_err_mean[NOSE_INDEX], 'Max:', l2_pixel_err_max[NOSE_INDEX])

        print('LEFT_EAR Pixel Error:          ', l2_pixel_err_mean[LEFT_EAR_INDEX], 'Max:', l2_pixel_err_max[LEFT_EAR_INDEX])
        print('RIGHT_EAR Pixel Error:         ', l2_pixel_err_mean[RIGHT_EAR_INDEX], 'Max:', l2_pixel_err_max[RIGHT_EAR_INDEX])

        print('BASE_NECK Pixel Error:         ', l2_pixel_err_mean[BASE_NECK_INDEX], 'Max:', l2_pixel_err_max[BASE_NECK_INDEX])

        print('LEFT_FRONT_PAW Pixel Error:    ', l2_pixel_err_mean[LEFT_FRONT_PAW_INDEX], 'Max:', l2_pixel_err_max[LEFT_FRONT_PAW_INDEX])
        print('RIGHT_FRONT_PAW Pixel Error:   ', l2_pixel_err_mean[RIGHT_FRONT_PAW_INDEX], 'Max:', l2_pixel_err_max[RIGHT_FRONT_PAW_INDEX])

        print('CENTER_SPINE Pixel Error:      ', l2_pixel_err_mean[CENTER_SPINE_INDEX], 'Max:', l2_pixel_err_max[CENTER_SPINE_INDEX])

        print('LEFT_REAR_PAW Pixel Error:     ', l2_pixel_err_mean[LEFT_REAR_PAW_INDEX], 'Max:', l2_pixel_err_max[LEFT_REAR_PAW_INDEX])
        print('RIGHT_REAR_PAW Pixel Error:    ', l2_pixel_err_mean[RIGHT_REAR_PAW_INDEX], 'Max:', l2_pixel_err_max[RIGHT_REAR_PAW_INDEX])

        print('BASE_TAIL Pixel Error:         ', l2_pixel_err_mean[BASE_TAIL_INDEX], 'Max:', l2_pixel_err_max[BASE_TAIL_INDEX])
        print('MID_TAIL Pixel Error:          ', l2_pixel_err_mean[MID_TAIL_INDEX], 'Max:', l2_pixel_err_max[MID_TAIL_INDEX])
        print('TIP_TAIL Pixel Error:          ', l2_pixel_err_mean[TIP_TAIL_INDEX], 'Max:', l2_pixel_err_max[TIP_TAIL_INDEX])


if __name__ == "__main__":
    main()
