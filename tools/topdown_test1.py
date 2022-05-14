#%%

# 1- load an annotated image
# 2- mask the segmentated image
# 3- get the keypoints


import itertools
import os
import sys
import time

import cv2
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
    
# paths
pkl_file = '/home/ghanba/mousepose_abed/scrap/all3_cloudfactory.h5'
dataset_path = '/projects/compsci/USERS/ghanba/cloudfactory_annotations/all_frames/'

cfg_file = '/projects/kumar-lab/multimouse-pipeline/model_zoo/hrnets/2021/multimouse_2021-10-26.yaml'
model_path = '/projects/kumar-lab/multimouse-pipeline/model_zoo/hrnets/2021/best_state.pth'

# model_name = 'multimouse_cloudfactory_10'
# model_path = f'/home/ghanba/mousepose_abed/deep-hres-net/output-multi-mouse/multimousepose/pose_hrnet/{model_name}/best_state.pth'
# cfg_file = f'/home/ghanba/mousepose_abed/deep-hres-net/experiments/multimouse/cloudfactory/{model_name}.yml'

# load training data
pose_labels = list(parse_poses_pkl(pkl_file))

# randomly select a traning image
np.random.seed(32)
f_idx = np.random.randint(0, len(pose_labels))
image_path = os.path.join(dataset_path, pose_labels[f_idx]['image_name'])
data_numpy = skimage.io.imread(image_path, as_gray=True) * 255

# show the image
plt.imshow(data_numpy, cmap='gray')
plt.show()

# get one of the masks
segs = pose_labels[f_idx]['seg_instances']
mask = get_mask([segs[1]], data_numpy.shape)

# show the mask
masked_img = (255-mask)+((mask>0).astype(int)*data_numpy).astype(int)
plt.imshow(masked_img, cmap='gray', vmin=0, vmax=100)
plt.show()

# make rgb image from masked_img
masked_img = np.repeat(masked_img[:, :, np.newaxis], 3, axis=2).astype(np.float32)
# show the rgb image
plt.imshow(masked_img)
plt.show()

# load config
cfg.defrost()
cfg.merge_from_file(cfg_file)

# cudnn related setting
cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
    cfg, is_train=False
)

# check if cuda is available
if torch.cuda.is_available():
    print('cuda is available')

# load model
model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
model.eval().cuda()

xform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.45, 0.45, 0.45],
            std=[0.225, 0.225, 0.225],
        ),
    ])

data_numpy = skimage.io.imread(image_path) * 255
tensor_stack = torch.stack([xform(data_numpy)])
# tensor_stack = torch.stack([xform(masked_img)])

start = time.time()
with torch.no_grad():
    output = model(tensor_stack.cuda())
end = time.time()
print('time taken: ', end - start)
output_arr = output.detach().cpu().numpy()

# get the keypoints
plt.imshow(output_arr[0, 6, :, :], cmap='cool')
plt.show()

# plot histogram of values in the output passed from a sigmoid layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
plt.hist(sigmoid(output_arr[0, 6, :, :].flatten()), bins=100)
plt.show()

# find embedding for the keypoints
min_pose_heatmap_val = .1
min_embed_sep = 0.2
max_embed_sep = 0.5
max_inst_dist = 100
min_joint_count = 5
max_instance_count = 10

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

# show image with keypoints
cmap = plt.get_cmap('Paired')
colors = [cmap(i) for i in np.linspace(0, 1, len(pose_labels[f_idx]['pose_instances']))] 
colors = itertools.cycle(colors)
plt.figure(figsize=(8,8))
plt.imshow(skimage.io.imread(image_path, as_gray=True) * 255, cmap='gray')
for i in range(points.shape[0]):
    color = next(colors)
    plt.scatter(points[i,:,1], points[i,:,0], s=8, color=color)
    #plot lines between points same color as points
    for j in range(points.shape[1]-1):
        if np.all(points[i,j,:] != [0,0]) and np.all(points[i,j+1,:] != [0,0]):
            plt.plot([points[i,j,1], points[i,j+1,1]], [points[i,j,0], points[i,j+1,0]], color=color)
plt.show()


# %%
