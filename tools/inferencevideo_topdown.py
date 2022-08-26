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

import time, datetime
import multiprocessing as mp

# Removes padding from a contour
def get_trimmed_contour(padded_contour, default_val=-1):
    mask = np.all(padded_contour==default_val, axis=1)
    trimmed_contour = np.reshape(padded_contour[~mask,:], [-1,2])
    return trimmed_contour.astype(np.int32)

# Helper function to return a contour list
# Returns a stack of length 1 if only 1 contour was stored
# Otherwise, returns the entire stack of contours
def get_contour_stack(contour_mat, default_val=-1):
    # Only one contour was stored per-mouse
    if np.ndim(contour_mat)==2:
        trimmed_contour = get_trimmed_contour(contour_mat, default_val)
        contour_stack = [trimmed_contour]
    # Entire contour list was stored
    elif np.ndim(contour_mat)==3:
        contour_stack = []
        for part_idx in np.arange(np.shape(contour_mat)[0]):
            cur_contour = contour_mat[part_idx]
            if np.all(cur_contour==default_val):
                break
            trimmed_contour = get_trimmed_contour(cur_contour, default_val)
            contour_stack.append(trimmed_contour)
    return contour_stack

def get_mask(contours, img_shape):
    # a function to get a numpy array of contours 
    # and return mask with all the inside pixels set to 1
    mask = np.zeros(img_shape, dtype=np.uint8)
    contour_stack = get_contour_stack(contours)
    _ = cv2.drawContours(mask, contour_stack, -1, (1), thickness=cv2.FILLED)
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

# Data controller that manages the reading, pre-processing, and post-processing of batches using multiple threads
# Note that if "frames" is specified, the resulting pose matrix will be of len(frames) that exists in the video
class data_controller:
    def __init__(self, video_path, seg_data_file, args, frames=None, preproc_thread_count=1):
        self.__video_path = video_path
        self.__seg_file = seg_data_file
        self.__args = args
        self.__frames = self.__safe_detect_frames(frames)
        #self.video_reader_queue = mp.Queue(self.__args.batch_size*preproc_thread_count*5)
        self.video_reader_queue = mp.Queue(preproc_thread_count*5)
        self.__reader_threads = None
        self.__preproc_thread_count = preproc_thread_count
        #self.results_queue = mp.Queue(self.__args.batch_size*5)
        self.results_queue = mp.Queue(5)
        self.__results_storage_thread = None
        self.__final_mat_queue = mp.JoinableQueue(1)
        self.__result_pose_mat = None
        self.__start_batch_generator()
        self.__start_dequeue_results()
    # Opens the video and returns a reader object
    def __open_video(self):
        return pims.Video(self.__video_path)
    def get_vid_len(self):
        vid = self.__open_video()
        vid_len = len(vid)
        vid.close()
        return vid_len
    def get_frames(self):
        return self.__frames
    # Attempts to detect all the frames that should be read
    def __safe_detect_frames(self, selected_frames):
        vid_len = self.get_vid_len()
        if selected_frames is None:
            return np.arange(vid_len)
        else:
            possible_frames = np.intersect_1d(np.arange(vid_len), selected_frames)
            if len(possible_frames)>0:
                return possible_frames
            else:
                raise Error('Selected frames don\'t exist in the video')
    # Input reading function
    def __read_batches(self, reader_queue, frames_to_read):
        vid = self.__open_video()
        seg_file = h5py.File(self.__seg_file, 'r')
        for frame in frames_to_read:
            seg_data = seg_file['poseest/seg_data'][frame,...]
            data_numpy = vid[frame]
            data_numpy = as_grey(data_numpy)[..., np.newaxis]
            img = data_numpy.copy()
            batch = []
            for i in np.arange(seg_data.shape[0]):
                # No segmentation stored here
                if np.all(seg_data[i,:]==-1):
                    continue
                mask = get_mask(seg_data[i,:], img.shape)
                data_numpy = (255-mask)+((mask>0).astype(int)*img).astype(np.float32)
                # Normalize function taken from xform
                data_numpy = np.repeat(data_numpy, 3, axis=2)
                data_numpy = np.transpose((data_numpy/255.-0.45)/0.225, [2,0,1])
                batch.append(data_numpy)
            if len(batch)!=0:
                reader_queue.put((frame, np.stack(batch)))
        vid.close()
        seg_file.close()
    def __start_batch_generator(self):
        if self.__reader_threads is None:
            self.__reader_threads = [mp.Process(target=self.__read_batches, args=(self.video_reader_queue, x,)) for x in np.array_split(self.__frames, self.__preproc_thread_count)]
            for thread in self.__reader_threads:
                thread.start()
    # Output receiver function
    def __dequeue_results(self, results_queue, return_queue):
        points = np.zeros((len(self.__frames), 1, 12, 2), dtype=np.uint16)
        for _ in self.__frames:
            frame, output = results_queue.get()
            # Resize output matrix if necessary
            if points.shape[1] < output.shape[0]:
                points_resized = np.zeros((len(self.__frames), output.shape[0], 12, 2), dtype=np.uint16)
                points_resized[:, :points.shape[1], ...] = points
                points = points_resized
            # Add new data into matrix
            frame_idx = np.where(frame == self.__frames)[0]
            points[frame_idx, :output.shape[0], ...] = output
        # Place the results in the joinable queue to be retrieved from this thread
        return_queue.put(points)
    def __start_dequeue_results(self):
        if self.__results_storage_thread is None:
            self.__results_storage_thread = mp.Process(target=self.__dequeue_results, args=(self.results_queue, self.__final_mat_queue))
            self.__results_storage_thread.start()
    # Safely retrieves the output when ready
    def get_poses(self):
        # Close reader threads
        if self.__reader_threads is not None:
            for thread in self.__reader_threads:
                thread.join()
            self.__reader_threads = None
        # Close postproc threads
        if self.__results_storage_thread is not None:
            self.__result_pose_mat = self.__final_mat_queue.get()
            self.__results_storage_thread.join()
            self.__results_storage_thread = None
        return self.__result_pose_mat


def main():
    parser = argparse.ArgumentParser(description='Pose Estimation Using Top Down Approach')
    parser.add_argument('--video_path', type=str, default='', help='video file path')
    parser.add_argument('--model_path', type=str, default='', help='model file path')
    parser.add_argument('--cfg_path', type=str, default='', help='cfg file path')
    parser.add_argument('--h5_path', type=str, default='', help='input file path')
    parser.add_argument('--h5_path_out', type=str, default='', help='output file path')
    parser.add_argument('--num_reader_threads', type=int, default=1, help='number of threads to read in data')
    parser.add_argument('--min_pose_heatmap_val', type=float, default=0.2, help='min pose heatmap val')
    parser.add_argument('--min_embed_sep', type=float, default=0.2, help='min embed sep')
    parser.add_argument('--max_embed_sep', type=float, default=0.4, help='max embed sep')
    parser.add_argument('--max_inst_dist', type=int, default=150, help='max instance dist')
    parser.add_argument('--min_joint_count', type=int, default=4, help='min joint count')
    parser.add_argument('--max_instance_count', type=int, default=3, help='max instance count')
    args = parser.parse_args()

    # copy the data to the output file
    with h5py.File(args.h5_path, 'r') as h5r:
        with h5py.File(args.h5_path_out, 'w') as h5w:
            for obj in h5r.keys():        
                h5r.copy(obj, h5w )

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

    # start up loader and post-processor queues
    controller = data_controller(args.video_path, args.h5_path_out, args, preproc_thread_count=args.num_reader_threads)
    print('video frame count: ' + str(len(controller.get_frames())))
    for _ in tqdm(controller.get_frames()):
    #for _ in controller.get_frames():
        # start_time = time.time()
        batch_frame, batch = controller.video_reader_queue.get()
        # delta_time = time.time()-start_time
        # print('Batch ' + str(batch_frame) + ' retrieved in ' + str(datetime.timedelta(seconds=delta_time)))
        # start_time = time.time()
        tensor_stack = torch.from_numpy(batch)
        # delta_time = time.time()-start_time
        # print('Batch ' + str(batch_frame) + ' stacked in ' + str(datetime.timedelta(seconds=delta_time)))
        # start_time = time.time()
        with torch.no_grad():
            output = model(tensor_stack.cuda())
            # delta_time = time.time()-start_time
            # print('Batch ' + str(batch_frame) + ' processed in ' + str(datetime.timedelta(seconds=delta_time)))
            # start_time = time.time()
            results_npy = []
            for j in range(output.shape[0]):
                results_npy.append(get_keypoints(output[j:j+1,...], cfg, 
                    args.min_pose_heatmap_val,
                    args.min_embed_sep,
                    args.max_embed_sep,
                    args.max_inst_dist,
                    args.min_joint_count,
                    args.max_instance_count,
                    )[0].astype(np.int16))
        # delta_time = time.time()-start_time
        # print('Batch ' + str(batch_frame) + ' (shape: ' + str(output.shape) + ') post-processed in ' + str(datetime.timedelta(seconds=delta_time)))
        # start_time = time.time()
        controller.results_queue.put((batch_frame, np.stack(results_npy)))
        # delta_time = time.time()-start_time
        # print('Batch ' + str(batch_frame) + ' placed in out_queue in ' + str(datetime.timedelta(seconds=delta_time)))
    points = controller.get_poses()
    with h5py.File(args.h5_path_out, 'r+') as h5file:
        h5file.create_dataset('poseest/points', data=points)
        

if __name__ == '__main__':
    main()
