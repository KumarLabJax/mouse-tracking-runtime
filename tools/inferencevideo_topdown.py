#%%
import argparse
import itertools
import os

import _init_paths

import cv2
import h5py
import matplotlib.pyplot as plt
import models
import numpy as np
import pandas as pd
import pims
import skimage.io
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import utils.assocembedutil as aeutil
from config import cfg
from tqdm import tqdm


def _read_image(image_path):
    data_numpy = skimage.io.imread(image_path, as_gray=True) * 255

    data_numpy = data_numpy.round().astype(np.uint8)
    data_numpy = data_numpy[..., np.newaxis]

    return data_numpy
def get_mask(contours, img_shape):
    # a function to get a numpy array of contours 
    # and return mask with all the inside pixels set to 1
    mask = np.zeros(img_shape, dtype=np.uint8)
    for cnt in contours:
        cv2.drawContours(mask, [cnt.astype(int)], 0, 255, -1)
    return mask

def overlay_skeleton(img, pose_instances, branches=None, title=None, add_image=False, colors=None):
    if branches is None:
        branches = [[4,6,5],[7,9,8],[0,3,6,9,10,11]]
    
    if add_image:
        plt.imshow(img, cmap='gray')
    cmap = plt.get_cmap('Paired')
    if colors is None:
        colors = [cmap(i) for i in np.linspace(0, 1, len(pose_instances))]
    colors_iter = itertools.cycle(colors)
    for pose in pose_instances:
        color = next(colors_iter)
        plt.scatter(pose[:,1], pose[:,0], s=8, color=color)
        #plot lines between points same color as points
        for branch in branches:
            for i in range(len(branch)-1):
                if np.all(pose[branch[i],:] != [0,0]) and np.all(pose[branch[i+1],:] != [0,0]):
                    plt.plot([pose[branch[i],1], pose[branch[i+1],1]], [pose[branch[i],0], pose[branch[i+1],0]], color=color)
        plt.title(title)
        plt.axis('off')

def get_keypoints(
            output, cfg, 
            min_pose_heatmap_val,
            min_embed_sep,
            max_embed_sep,
            max_inst_dist,
            min_joint_count,
            max_instance_count,
            ):
    pose_heatmap = output[0,:cfg.MODEL.NUM_JOINTS,:,:]
    pose_localmax = aeutil.localmax2D(pose_heatmap, min_pose_heatmap_val, 3)

    pose_instances = aeutil.calc_pose_instances(
                                pose_heatmap,
                                pose_localmax,
                                pose_heatmap,
                                min_embed_sep,
                                max_embed_sep,
                                max_inst_dist)
    
    if min_joint_count is not None:
        pose_instances = [
            pi for pi in pose_instances
            if len(pi.keypoints) >= min_joint_count
        ]
    
    # if we have too many poses remove in order of lowest confidence
    if (max_instance_count is not None
            and len(pose_instances) > max_instance_count):
        pose_instances.sort(key=lambda pi: pi.mean_inst_conf)
        del pose_instances[max_instance_count:]
    
    points = np.zeros(
                (max_instance_count, 12, 2),
                dtype=np.uint16)
    for pose_index, pose_instance in enumerate(pose_instances):
        for keypoint in pose_instance.keypoints.values():
            points[pose_index, keypoint['joint_index'], 0] = keypoint['y_pos']
            points[pose_index, keypoint['joint_index'], 1] = keypoint['x_pos']
    
    return points

def as_grey(frame):
    red = frame[:, :, 0]
    green = frame[:, :, 1]
    blue = frame[:, :, 2]
    return 0.2125 * red + 0.7154 * green + 0.0721 * blue

def main():
    parser = argparse.ArgumentParser(description='Pose Estimation Using Top Down Approach')
    parser.add_argument('--video_path', type=str, default='', help='video file path')
    parser.add_argument('--model_path', type=str, default='', help='model file path')
    parser.add_argument('--cfg_path', type=str, default='', help='cfg file path')
    parser.add_argument('--h5_path', type=str, default='', help='input file path')
    parser.add_argument('--h5_path_out', type=str, default='', help='output file path')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--min_pose_heatmap_val', type=float, default=0.2, help='min pose heatmap val')
    parser.add_argument('--min_embed_sep', type=float, default=0.2, help='min embed sep')
    parser.add_argument('--max_embed_sep', type=float, default=0.4, help='max embed sep')
    parser.add_argument('--max_inst_dist', type=int, default=150, help='max instance dist')
    parser.add_argument('--min_joint_count', type=int, default=4, help='min joint count')
    parser.add_argument('--max_instance_count', type=int, default=3, help='max instance count')
    args = parser.parse_args()

    vid = pims.Video(args.video_path)
    vid_len = len(vid)
    print('video length: {}'.format(vid_len))

    h5r=h5py.File(args.h5_path, 'r')
    with h5py.File(args.h5_path_out, 'w') as h5w:
        for obj in h5r.keys():        
            h5r.copy(obj, h5w )
    h5r.close()

    pose_est = h5py.File(args.h5_path, 'r')
    seg_data = pose_est['poseest']['seg_data']

    # load config
    cfg.defrost()
    cfg.merge_from_file(args.cfg_path)
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    # load model
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=False)
    model.eval().cuda()

    xform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.45, 0.45, 0.45],
                std=[0.225, 0.225, 0.225],
            ),
        ])

    points = np.zeros((vid_len, 4, 12, 2), dtype=np.uint16)
    for frame_num in tqdm(range(vid_len//1000)):
        
        pose_est_data = seg_data[frame_num,...]

        data_numpy = vid[frame_num]
        data_numpy = as_grey(data_numpy)[..., np.newaxis]
        img = data_numpy.copy()
        batch = []
        for seg in pose_est_data:
            # assumes continuos segmentation array with later indeces being zero
            seg = seg[seg!=-1].reshape(-1,2)
            if seg.shape[0]<3:
                continue

            mask = get_mask([seg], img.shape)
            data_numpy = (255-mask)+((mask>0).astype(int)*img).astype(np.uint8)
            batch.append(np.repeat(data_numpy, 3, axis=2))

        tensor_stack = torch.stack([xform(d) for d in batch])
        with torch.no_grad():
            output = model(tensor_stack.cuda())

        start_time = time.time()
        for j in range(output.shape[0]):
                points[frame_num, j, ...] = get_keypoints(output[j:j+1,...], cfg, 
                    args.min_pose_heatmap_val,
                    args.min_embed_sep,
                    args.max_embed_sep,
                    args.max_inst_dist,
                    args.min_joint_count,
                    args.max_instance_count,
                    )[0].astype(np.int16)
        print('time: {}'.format(time.time()-start_time))
    with h5py.File(args.h5_path_out, 'r+') as h5file:
        data = h5file['poseest']['points']
        data[...] = points

if __name__ == '__main__':
    '''example:
        video_path='/home/ghanba/mousepose_abed/data/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1.avi'
        model_path='/home/ghanba/mousepose_abed/deep-hres-net/output-multi-mouse-topdown/multimousepose/pose_hrnet/multimouse_topdown_1/best_state.pth'
        cfg_path='/home/ghanba/mousepose_abed/deep-hres-net/experiments/multimouse/cloudfactory/multimouse_topdown_1.yml'
        h5_path='/home/ghanba/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1_pose_est_v4.h5'
        h5_path_out='/home/ghanba/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1_pose_est_v4_out.h5'


    # python3 inferencevide0_topdown.py \
    #     --video_path=/home/ghanba/mousepose_abed/data/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1.avi \
    #     --model_path=/home/ghanba/mousepose_abed/deep-hres-net/output-multi-mouse-topdown/multimousepose/pose_hrnet/multimouse_topdown_1/best_state.pth \
    #     --cfg_path=/home/ghanba/mousepose_abed/deep-hres-net/experiments/multimouse/cloudfactory/multimouse_topdown_1.yml \
    #     --h5_path=/projects/kumar-lab/bgeuther/deeplab/Brians_PrototypingTraining/testing_2022-04-19/poses_2021/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1_pose_est_v4.h5 \

    python inferencevideo_topdown.py  \
        --video_path=/home/ghanba/mousepose_abed/data/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1.avi         \
        --model_path=/home/ghanba/mousepose_abed/deep-hres-net/output-multi-mouse-topdown/multimousepose/pose_hrnet/multimouse_topdown_1/best_state.pth         \
        --cfg_path=/home/ghanba/mousepose_abed/deep-hres-net/experiments/multimouse/cloudfactory/multimouse_topdown_1.yml         \
        --h5_path=/home/ghanba/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1_pose_est_v4.h5 \
        --h5_path_out=/home/ghanba/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1_pose_est_v4_out.h5 

    python -u tools/rendervidoverlay.py \
        --exclude-forepaws --exclude-ears \
        vid --in-vid '/home/ghanba/mousepose_abed/data/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1.avi' \
        --in-pose '/home/ghanba/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1_pose_est_v4_out.h5' \
        --out-vid '/home/ghanba/mousepose_abed/data/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1_topdown.avi'

    '''
    # main()
