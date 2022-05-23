#%%

# 1- load an annotated image
# 2- mask the segmentated image
# 3- get the keypoints


import itertools
import os
import sys
import time

import cv2
import pims
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io

import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

# load deep-hres-net modules
module_path = './lib'
if module_path not in sys.path:
    sys.path.append(module_path)

import models
import utils.assocembedutil as aeutil
from config import cfg


# helper functions
def parse_poses_pkl(pkl_file):
    df = pd.read_hdf(pkl_file, key='df')
    # make a generator that yields the image name and 
    # the pose_instances and seg_instances
    for i, row in df.iterrows():
        yield {
            'image_name': row['image_name'],
            'pose_instances': row['pose_instances'],
            'seg_instances': row['seg_instances'],
        }

def get_mask(contours, img_shape):
    # a function to get a numpy array of contours 
    # and return mask with all the inside pixels set to 1
    mask = np.zeros(img_shape, dtype=np.uint8)
    for cnt in contours:
        cv2.drawContours(mask, [cnt.astype(int)], 0, 255, -1)
    return mask


def overlay_skeleton(img, pose_instances, branches=None, title=None):
    if branches is None:
        branches = [[4,6,5],[7,9,8],[0,3,6,9,10,11]]
    
    plt.imshow(img, cmap='gray')
    cmap = plt.get_cmap('Paired')
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


def get_embedding(data_numpy, model_path, cfg_file):
    # load config
    cfg.defrost()
    cfg.merge_from_file(cfg_file)
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    # load model
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.eval().cuda()

    xform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.45, 0.45, 0.45],
                std=[0.225, 0.225, 0.225],
            ),
        ])

    tensor_stack = torch.stack([xform(data_numpy)])


    with torch.no_grad():
        output = model(tensor_stack.cuda())
    # output_arr = output.detach().cpu().numpy()
    return output, cfg

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
    pose_localmax = aeutil.localmax2D(pose_heatmap, min_pose_heatmap_val, 5)
    pose_embed_map = output[0,cfg.MODEL.NUM_JOINTS:,:,:]
    pose_instances = aeutil.calc_pose_instances(
                                pose_heatmap,
                                pose_localmax,
                                pose_embed_map,
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


# paths
pkl_file = '/home/ghanba/mousepose_abed/scrap/all3_cloudfactory.h5'
dataset_path = '/projects/compsci/USERS/ghanba/cloudfactory_annotations/all_frames/'

cfg_file1 = '/projects/kumar-lab/multimouse-pipeline/model_zoo/hrnets/2021/multimouse_2021-10-26.yaml'
model_path1 = '/projects/kumar-lab/multimouse-pipeline/model_zoo/hrnets/2021/best_state.pth'

model_name = 'multimouse_cloudfactory'
model_path2 = f'/home/ghanba/mousepose_abed/deep-hres-net/output-multi-mouse1/multimousepose/pose_hrnet/{model_name}/best_state.pth'
cfg_file2 = f'/home/ghanba/mousepose_abed/deep-hres-net/tmp/cloudfactory/{model_name}.yml'

video_path = '/home/ghanba/mousepose_abed/data/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1.avi'
frame_num = 17629

# find embedding for the keypoints
min_pose_heatmap_val = .2
min_embed_sep = 0.2
max_embed_sep = 0.4
max_inst_dist = 150
min_joint_count = 3
max_instance_count = 10

vid = pims.Video(video_path)
data_numpy = vid[frame_num]

output, cfg = get_embedding(data_numpy, model_path1, cfg_file1)
points1 = get_keypoints(output, cfg, min_pose_heatmap_val, min_embed_sep, max_embed_sep, max_inst_dist, min_joint_count, max_instance_count)

output, cfg = get_embedding(data_numpy, model_path2, cfg_file2)
points2 = get_keypoints(output, cfg, min_pose_heatmap_val, min_embed_sep, max_embed_sep, max_inst_dist, min_joint_count, max_instance_count)

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
overlay_skeleton(data_numpy, points, title='pose_instances')

#%%
# get pose 
'''
python -u tools/infermultimousepose.py \
    --max-instance-count 4 \
    --max-embed-sep-within-instances 0.4 \
    --min-embed-sep-between-instances 0.2 \
    --min-pose-heatmap-val .2 \
    /home/ghanba/mousepose_abed/deep-hres-net/output-multi-mouse2/multimousepose/pose_hrnet/multimouse_cloudfactory/best_state.pth \
    /home/ghanba/mousepose_abed/deep-hres-net/tmp/cloudfactory/multimouse_cloudfactory.yml \
    '/home/ghanba/mousepose_abed/data/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1.avi' \
    '/home/ghanba/mousepose_abed/data/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1.h5'

python -u tools/rendervidoverlay.py \
    --exclude-forepaws --exclude-ears \
    vid --in-vid '/home/ghanba/mousepose_abed/data/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1.avi' \
    --in-pose '/home/ghanba/mousepose_abed/data/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1.h5' \
    --out-vid '/home/ghanba/mousepose_abed/data/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1_pose.avi'
'''
