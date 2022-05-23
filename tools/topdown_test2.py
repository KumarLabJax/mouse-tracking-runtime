#%%
import skimage.io
import sys
module_path = './lib'
if module_path not in sys.path:
    sys.path.append(module_path)

import itertools
import models
import utils.assocembedutil as aeutil
from config import cfg
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import os
import pims

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

    pose_embed_map = pose_heatmap
    print(pose_embed_map.shape)

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


#%%

pkl_file = '/home/ghanba/mousepose_abed/scrap/all3_cloudfactory.h5'
dataset_path = '/projects/compsci/USERS/ghanba/cloudfactory_annotations/all_frames/'

df = pd.read_hdf(pkl_file, key='df')


# %%

# randomly select a row from df
pose_label = df.sample(1).iloc[0]
image_path = os.path.join(dataset_path, pose_label['image_name'])

data_numpy = _read_image(image_path)
print(data_numpy.shape)
print(data_numpy.dtype)

plt.imshow(data_numpy, cmap='gray', vmin=0, vmax=255)
plt.show()


segs = pose_label['seg_instances']
rand_inst_idx = np.random.randint(0, len(segs))
mask = get_mask([segs[rand_inst_idx]], data_numpy.shape)
data_numpy = (255-mask)+((mask>0).astype(int)*data_numpy).astype(np.uint8)
print(data_numpy.shape)
print(data_numpy.dtype)

plt.imshow(data_numpy, cmap='gray', vmin=0, vmax=255)
plt.show()

# check output form model
cfg_file = '/home/ghanba/mousepose_abed/deep-hres-net/experiments/multimouse/cloudfactory/multimouse_topdown_1.yml'
model_path = '/home/ghanba/mousepose_abed/deep-hres-net/output-multi-mouse-topdown/multimousepose/pose_hrnet/multimouse_topdown_1/best_state.pth'

# find embedding for the keypoints
min_pose_heatmap_val = .2
min_embed_sep = 0.2
max_embed_sep = 0.4
max_inst_dist = 150
min_joint_count = 3
max_instance_count = 1

# replicate image to 3 channels
data_numpy = np.repeat(data_numpy, 3, axis=2)
output, cfg = get_embedding(data_numpy, model_path, cfg_file)
# stack torch output to output
output = torch.cat([output, output], dim=1)

points = get_keypoints(output, cfg, min_pose_heatmap_val, min_embed_sep, max_embed_sep, max_inst_dist, min_joint_count, max_instance_count)

plt.figure(figsize=(10,10))
overlay_skeleton(data_numpy, points, title='pose_instances', add_image=True)
cmap = plt.get_cmap('Paired')
colors = [cmap(i) for i in np.linspace(0, 1, 2)][1:]
overlay_skeleton(data_numpy, [pose_label['pose_instances'][rand_inst_idx][:,::-1]], title='pose_instances', colors=colors)

print(output.shape)
output = output[0,...].detach().cpu().numpy()
colored_output = np.stack([output[6,:,:], output[7,:,:], output[8,:,:]], axis=2)
print(colored_output.shape)
print(colored_output.dtype)
print(colored_output.max())
normalized_output = (colored_output - colored_output.min())/(colored_output.max()-colored_output.min())
print(normalized_output.max())

plt.figure(figsize=(10,10))
plt.imshow(normalized_output)
plt.show()
# %% load contours from file and do inference for few frames
video_path = '/home/ghanba/mousepose_abed/data/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1.avi'
frame_num = 10292

h5_path = '/projects/kumar-lab/bgeuther/deeplab/Brians_PrototypingTraining/testing_2022-04-19/poses_2021/B6J_3M_stranger_4day+NV10-CBAX2+2019-07-26+MDX0008_2019-07-26_16-00-00_1_pose_est_v4.h5'

import h5py

pose_est = h5py.File(h5_path, 'r')

pose_est_data = pose_est['poseest']['seg_data'][frame_num,...]

vid = pims.Video(video_path)
data_numpy = vid[frame_num]
data_numpy = as_grey(data_numpy)[..., np.newaxis]
img = data_numpy.copy()
plt.imshow(data_numpy, cmap='gray', vmin=0, vmax=255)
plt.show()

points = []
for seg in pose_est_data:
    seg = seg[seg!=-1].reshape(-1,2)
    if seg.shape[0]<3:
        continue

    mask = get_mask([seg], img.shape)
    data_numpy = (255-mask)+((mask>0).astype(int)*img).astype(np.uint8)


    # plt.imshow(data_numpy, cmap='gray', vmin=0, vmax=100)
    # plt.show()

    data_numpy1 = np.repeat(data_numpy, 3, axis=2)
    output, cfg = get_embedding(data_numpy1, model_path, cfg_file)
    # stack torch output to output
    output = torch.cat([output, output], dim=1)

    points.append(
        get_keypoints(output, cfg, min_pose_heatmap_val, 
        min_embed_sep, max_embed_sep, max_inst_dist, min_joint_count, 
        max_instance_count)[0]
        ) 

plt.figure(figsize=(10,10))
overlay_skeleton(img, points, title='pose_instances', add_image=True)

# %%
# create a randim tensoor
torch.random.manual_seed(0)
a = torch.rand(3,2,2) 
a
a = a - torch.tensor([1,2,3]).reshape(3,1,1)
a