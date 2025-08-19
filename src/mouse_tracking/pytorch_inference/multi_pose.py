"""Inference function for executing pytorch for a multi mouse pose model."""

import queue
import sys
import time

import h5py
import imageio
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from mouse_tracking.models.model_definitions import MULTI_MOUSE_POSE
from mouse_tracking.pytorch_inference.hrnet.config import cfg
from mouse_tracking.pytorch_inference.hrnet.models import pose_hrnet
from mouse_tracking.utils.hrnet import argmax_2d_torch, preprocess_hrnet
from mouse_tracking.utils.pose import render_pose_overlay
from mouse_tracking.utils.prediction_saver import prediction_saver
from mouse_tracking.utils.segmentation import get_frame_masks
from mouse_tracking.utils.timers import time_accumulator
from mouse_tracking.utils.writers import (
    adjust_pose_version,
    write_pose_v2_data,
    write_pose_v3_data,
)


def predict_pose_topdown(
    input_iter, mask_file, model, render: str | None = None, batch_size: int = 1
):
    """Main function that processes an iterator.

    Args:
            input_iter: an iterator that will produce frame inputs
            mask_file: kumar lab pose file containing segmentation data
            model: pytorch loaded model
            render: optional output file for rendering a prediction video
            batch_size: number of frames to predict per-batch

    Returns:
            tuple of (pose_out, conf_out, performance)
            pose_out: output accumulator for keypoint location data
            conf_out: output accumulator for confidence of keypoint data
            performance: timing performance logs
    """
    mask_file = h5py.File(mask_file, "r")
    if "poseest/seg_data" not in mask_file:
        raise ValueError(f"Segmentation not present in pose file {mask_file}.")

    pose_results = prediction_saver(dtype=np.uint16)
    confidence_results = prediction_saver(dtype=np.float32)

    if render is not None:
        vid_writer = imageio.get_writer(render, fps=30)

    performance_accumulator = time_accumulator(
        3, ["Preprocess", "GPU Compute", "Postprocess"], frame_per_batch=batch_size
    )

    # Main loop for inference
    video_done = False
    batch_num = 0
    frame_idx = 0
    while not video_done:
        t1 = time.time()
        # accumulator for unaltered frames
        full_frame_batch = []
        # accumulator for inputs to network
        mouse_batch = []
        # accumulator to indicate number of inputs per frame within the batch
        # [1, 3, 2] would indicate a total batch size of 6 that spans 3 frames
        # value indicates number of inputs and predictions to use per frame
        batch_frame_count = []
        batch_count = 0
        num_frames_in_batch = 0
        for _batch_frame_idx in np.arange(batch_size):
            try:
                input_frame = next(input_iter)
                full_frame_batch.append(input_frame)
                seg_data = mask_file["poseest/seg_data"][frame_idx, ...]
                masks_batch = get_frame_masks(seg_data, input_frame.shape[:2])
                masks_in_frame = 0
                for current_mask_idx in range(len(masks_batch)):
                    # Skip if no mask
                    if not np.any(masks_batch[current_mask_idx]):
                        continue
                    batch = (
                        np.repeat(255 - masks_batch[current_mask_idx], 3).reshape(
                            input_frame.shape
                        )
                        + (
                            np.repeat(masks_batch[current_mask_idx], 3).reshape(
                                input_frame.shape
                            )
                            * input_frame
                        )
                    ).astype(np.uint8)
                    mouse_batch.append(preprocess_hrnet(batch))
                    batch_count += 1
                    masks_in_frame += 1
                frame_idx += 1
                num_frames_in_batch += 1
                batch_frame_count.append(masks_in_frame)
            except StopIteration:
                video_done = True
                break

        # No masks, nothing to predict, go to next batch after providing default data
        if batch_count == 0:
            t2 = time.time()
            default_pose = np.full([num_frames_in_batch, 1, 12, 2], 0, np.int64)
            default_conf = np.full([num_frames_in_batch, 1, 12], 0, np.float32)
            pose_results.results_receiver_queue.put(
                (num_frames_in_batch, default_pose), timeout=5
            )
            confidence_results.results_receiver_queue.put(
                (num_frames_in_batch, default_conf), timeout=5
            )
            t4 = time.time()
            # compute skipped
            performance_accumulator.add_batch_times([t1, t2, t2, t4])
            continue

        batch_shape = [batch_count, 3, input_frame.shape[0], input_frame.shape[1]]
        batch_tensor = torch.empty(batch_shape, dtype=torch.float32)
        for i, frame in enumerate(mouse_batch):
            batch_tensor[i] = frame
        batch_num += 1

        t2 = time.time()
        with torch.no_grad():
            output = model(batch_tensor.cuda())
        t3 = time.time()
        confidence_cuda, pose_cuda = argmax_2d_torch(output)
        confidence = confidence_cuda.cpu().numpy()
        pose = pose_cuda.cpu().numpy()
        # disentangle batch -> frame data
        pose_stacked = np.full(
            [num_frames_in_batch, np.max(batch_frame_count), 12, 2], 0, np.int64
        )
        conf_stacked = np.full(
            [num_frames_in_batch, np.max(batch_frame_count), 12], 0, np.float32
        )
        cur_idx = 0
        for cur_frame_idx, num_obs in enumerate(batch_frame_count):
            if num_obs == 0:
                continue
            pose_stacked[cur_frame_idx, :num_obs] = pose[cur_idx : (cur_idx + num_obs)]
            conf_stacked[cur_frame_idx, :num_obs] = confidence[
                cur_idx : (cur_idx + num_obs)
            ]
            cur_idx += num_obs

        try:
            pose_results.results_receiver_queue.put(
                (num_frames_in_batch, pose_stacked), timeout=5
            )
            confidence_results.results_receiver_queue.put(
                (num_frames_in_batch, conf_stacked), timeout=5
            )
        except queue.Full:
            if not pose_results.is_healthy() or not confidence_results.is_healthy():
                print("Writer thread died unexpectedly.", file=sys.stderr)
                sys.exit(1)
            print(
                f"WARNING: Skipping inference on batch: {batch_num}, frames: {frame_idx - num_frames_in_batch}-{frame_idx - 1}"
            )
            continue
        if render is not None:
            for idx in np.arange(num_frames_in_batch):
                rendered_pose = full_frame_batch[idx].astype(np.uint8)
                for cur_frame_idx in np.arange(pose_stacked.shape[1]):
                    current_pose = pose_stacked[idx, cur_frame_idx]
                    current_confidence = conf_stacked[idx, cur_frame_idx]
                    rendered_pose = render_pose_overlay(
                        rendered_pose,
                        current_pose,
                        np.argwhere(current_confidence == 0).flatten(),
                    )
                vid_writer.append_data(rendered_pose)
        t4 = time.time()
        performance_accumulator.add_batch_times([t1, t2, t3, t4])

    pose_results.results_receiver_queue.put((None, None))
    confidence_results.results_receiver_queue.put((None, None))
    return (pose_results, confidence_results, performance_accumulator)


def infer_multi_pose_pytorch(args):
    """Main function to run a single mouse pose model."""
    model_definition = MULTI_MOUSE_POSE[args.model]
    cfg.defrost()
    cfg.merge_from_file(model_definition["pytorch-config"])
    cfg.TEST.MODEL_FILE = model_definition["pytorch-model"]
    cfg.freeze()
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    model = pose_hrnet.get_pose_net(cfg, is_train=False)
    model.load_state_dict(
        torch.load(cfg.TEST.MODEL_FILE, weights_only=True), strict=False
    )
    model.eval()
    model = model.cuda()

    if args.video:
        vid_reader = imageio.get_reader(args.video)
        frame_iter = vid_reader.iter_data()
    else:
        single_frame = imageio.imread(args.frame)
        frame_iter = [single_frame]

    pose_results, confidence_results, performance_accumulator = predict_pose_topdown(
        frame_iter, args.out_file, model, args.out_video, args.batch_size
    )
    pose_matrix = pose_results.get_results()
    confidence_matrix = confidence_results.get_results()
    write_pose_v2_data(
        args.out_file,
        pose_matrix,
        confidence_matrix,
        model_definition["model-name"],
        model_definition["model-checkpoint"],
    )
    # Make up fake data for v3 data...
    instance_count = np.sum(np.any(confidence_matrix > 0, axis=2), axis=1).astype(
        np.uint8
    )
    instance_embedding = np.full(confidence_matrix.shape, 0, dtype=np.float32)
    # TODO: Make a better dummy (low cost) tracklet generation or allow user to pick one...
    # This one essentially produces valid but horrible data (index means idenitity)
    instance_track_id = (
        np.tile([np.arange(confidence_matrix.shape[1])], confidence_matrix.shape[0])
        .reshape(confidence_matrix.shape[:2])
        .astype(np.uint32)
    )
    # instance_track_id = np.zeros(confidence_matrix.shape[:2], dtype=np.uint32)
    for row in range(len(instance_track_id)):
        valid_poses = instance_count[row]
        instance_track_id[row, instance_track_id[row] >= valid_poses] = 0
    write_pose_v3_data(
        args.out_file, instance_count, instance_embedding, instance_track_id
    )
    # Since this is topdown, segmentation is present and we can instruct it that it's there
    adjust_pose_version(args.out_file, 6)
    performance_accumulator.print_performance()
