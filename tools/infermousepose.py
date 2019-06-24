import argparse
import h5py
import imageio
import numpy as np
import time

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-file',
        help='the model file to use for inference',
        default=None,
    )

    parser.add_argument(
        'cfg',
        help='the configuration for the model to use for inference',
    )

    parser.add_argument(
        'video',
        help='the input video',
    )

    parser.add_argument(
        'poseout',
        help='the pose estimation output HDF5 file',
    )

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

    with torch.no_grad(), imageio.get_reader(args.video) as reader:

        all_preds = []
        all_maxvals = []
        batch = []
        def perform_inference():
            if batch:
                batch_tensor = torch.stack([xform(img) for img in batch]).cuda()
                batch.clear()

                inf_out = model(batch_tensor)
                in_out_ratio = batch_tensor.size(-1) // inf_out.size(-1)
                if in_out_ratio == 4:
                    inf_out = torchfunc.upsample(inf_out, scale_factor=4, mode='bicubic', align_corners=False)

                maxvals, preds = argmax_2d(inf_out)
                maxvals = maxvals.cpu().numpy()
                preds = preds.cpu().numpy().astype(np.uint16)

                all_preds.append(preds)
                all_maxvals.append(maxvals)

        for frame_index, image in enumerate(reader):

            if frame_index != 0 and frame_index % FRAMES_PER_MINUTE == 0:
                curr_time = time.time()
                cum_time_elapsed = curr_time - start_time
                print('processed {:.1f} min of video in {:.1f} min'.format(
                    frame_index / FRAMES_PER_MINUTE,
                    cum_time_elapsed / 60,
                ))

            batch.append(image)
            if len(batch) == cfg.TEST.BATCH_SIZE_PER_GPU:
                perform_inference()

        perform_inference()

        all_preds = np.concatenate(all_preds)
        all_maxvals = np.concatenate(all_maxvals)

        with h5py.File(args.poseout, 'w') as h5file:
            h5file['poseest/points'] = all_preds
            h5file['poseest/confidence'] = all_maxvals


if __name__ == "__main__":
    main()
