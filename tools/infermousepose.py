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
from core.inference import get_final_preds
from core.inference import get_max_preds

import models


FRAMES_PER_MINUTE = 30 * 60

# done = mp.Event()
# def frame_gen_proc(video, queue, max_frames=None):
#     xform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.45, 0.45, 0.45],
#             std=[0.225, 0.225, 0.225],
#         ),
#     ])

#     if max_frames is None or max_frames >= 1:
#         with imageio.get_reader(video) as reader:
#             for frame_index, image in enumerate(reader):
#                 image_tensor = xform(image)
#                 # print(image_tensor.dtype)
#                 # print('image_tensor.is_pinned():', image_tensor.is_pinned())
#                 # image_tensor = image_tensor.pin_memory()
#                 image_tensor = image_tensor.cuda()
#                 queue.put(image_tensor)
#                 if max_frames is not None and frame_index >= max_frames - 1:
#                     break

#     # done token
#     queue.put(None)
#     done.wait()


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
                inf_out = torchfunc.upsample(inf_out, scale_factor=4, mode='bicubic', align_corners=False)
                inf_out = inf_out.cpu().numpy()

                preds, maxvals = get_max_preds(inf_out)
                preds = preds.astype(np.uint16)
                maxvals = maxvals.squeeze(2)

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
            h5file['poseest/points'] = all_preds[:, :, [1, 0]]
            h5file['poseest/confidence'] = all_maxvals

    # with torch.no_grad():

    #     all_preds = []
    #     all_maxvals = []
    #     batch = []
    #     def perform_inference():
    #         if batch:
    #             # batch_tensor = torch.stack([xform(img) for img in batch]).cuda()
    #             batch_tensor = torch.stack(batch).cuda()
    #             batch.clear()

    #             inf_out = model(batch_tensor)
    #             inf_out = torchfunc.upsample(inf_out, scale_factor=4, mode='bicubic', align_corners=False)
    #             inf_out = inf_out.cpu().numpy()

    #             preds, maxvals = get_max_preds(inf_out)
    #             preds = preds.astype(np.uint16)
    #             maxvals = maxvals.squeeze(2)

    #             all_preds.append(preds)
    #             all_maxvals.append(maxvals)

    #     queue = mp.Queue(cfg.TEST.BATCH_SIZE_PER_GPU * 2)
    #     frame_gen_p = mp.Process(target=frame_gen_proc, args=(args.video, queue, 5000))
    #     frame_gen_p.start()

    #     frame_index = 0
    #     while True:

    #         image_tensor = queue.get()
    #         if image_tensor is None:
    #             break

    #         if frame_index != 0 and frame_index % FRAMES_PER_MINUTE == 0:
    #             curr_time = time.time()
    #             cum_time_elapsed = curr_time - start_time
    #             print('processed {:.1f} min of video in {:1f} min'.format(
    #                 frame_index / FRAMES_PER_MINUTE,
    #                 cum_time_elapsed / 60,
    #             ))

    #         batch.append(image_tensor)
    #         if len(batch) == cfg.TEST.BATCH_SIZE_PER_GPU:
    #             perform_inference()

    #         frame_index += 1

    #     perform_inference()

    #     print('inference complete, saving results')

    #     all_preds = np.concatenate(all_preds)
    #     all_maxvals = np.concatenate(all_maxvals)

    #     with h5py.File(args.poseout, 'w') as h5file:
    #         h5file['poseest/points'] = all_preds[:, :, [1, 0]]
    #         h5file['poseest/confidence'] = all_maxvals

    #     print('cleaning up')
    #     done.set()
    #     frame_gen_p.join()

if __name__ == "__main__":
    main()
