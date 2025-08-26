"""Inference function for executing pytorch for a fecal boli detection model."""

import queue
import sys
import time

import imageio
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from mouse_tracking.models.model_definitions import FECAL_BOLI
from mouse_tracking.pytorch_inference.hrnet.config import cfg
from mouse_tracking.pytorch_inference.hrnet.models import pose_hrnet
from mouse_tracking.utils.arrays import get_peak_coords
from mouse_tracking.utils.hrnet import localmax_2d_torch, preprocess_hrnet
from mouse_tracking.utils.prediction_saver import prediction_saver
from mouse_tracking.utils.static_objects import plot_keypoints
from mouse_tracking.utils.timers import time_accumulator
from mouse_tracking.utils.writers import write_fecal_boli_data


def predict_fecal_boli(
    input_iter,
    model,
    render: str | None = None,
    frame_interval: int = 1,
    batch_size: int = 1,
):
    """Main function that processes an iterator.

    Args:
            input_iter: an iterator that will produce frame inputs
            model: pytorch loaded model
            render: optional output file for rendering a prediction video
            frame_interval: interval of frames to make predictions on
            batch_size: number of frames to predict per-batch

    Returns:
            tuple of (fecal_boli_out, count_out, performance)
            fecal_boli_out: output accumulator for keypoint location data
            count_out: output accumulator for counts
            performance: timing performance logs
    """
    fecal_boli_results = prediction_saver(dtype=np.uint16)
    fecal_boli_counts = prediction_saver(dtype=np.uint16)

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
        batch = []
        batch_count = 0
        for _ in np.arange(batch_size):
            try:
                while True:
                    input_frame = next(input_iter)
                    frame_idx += 1
                    if frame_idx % frame_interval == 0:
                        break
                batch.append(input_frame)
                batch_count += 1
                frame_idx += 1
            except StopIteration:
                video_done = True
                break
        if batch_count == 0:
            video_done = True
            break
        # concatenate will squeeze batch dim if it is of size 1, so only concat if > 1
        elif batch_count == 1:
            batch_tensor = preprocess_hrnet(batch[0])
        elif batch_count > 1:
            batch_tensor = torch.concatenate([preprocess_hrnet(x) for x in batch])
        batch_num += 1

        t2 = time.time()
        with torch.no_grad():
            output = model(batch_tensor.cuda())
        t3 = time.time()
        # These values were optimized for peakfinding for the 2020 fecal boli model and should not be modified
        # TODO:
        # Move these values to be attached to a specific model
        peaks_cuda = localmax_2d_torch(output, 0.75, 5)
        peaks = peaks_cuda.cpu().numpy()
        for batch_idx in np.arange(batch_count):
            _, new_coordinates = get_peak_coords(peaks[batch_idx][0])
            if len(new_coordinates) == 0:
                boli_coordinates = np.zeros([1, 0, 2], dtype=np.uint16)
                num_boli = np.array(0, dtype=np.uint16).reshape([1, -1])
            else:
                boli_coordinates = np.expand_dims(np.asarray(new_coordinates), axis=0)
                num_boli = np.array(boli_coordinates.shape[1], dtype=np.uint16).reshape(
                    [1, -1]
                )

            try:
                fecal_boli_results.results_receiver_queue.put(
                    (1, boli_coordinates), timeout=5
                )
                fecal_boli_counts.results_receiver_queue.put((1, num_boli), timeout=5)
            except queue.Full:
                if (
                    not fecal_boli_results.is_healthy()
                    or not fecal_boli_counts.is_healthy()
                ):
                    print("Writer thread died unexpectedly.", file=sys.stderr)
                    sys.exit(1)
                print(
                    f"WARNING: Skipping inference on batch: {batch_num}, frame: {batch_num * batch_size}"
                )
                continue
            if render is not None:
                rendered_keypoints = plot_keypoints(
                    new_coordinates, batch[batch_idx].astype(np.uint8), is_yx=True
                )
                vid_writer.append_data(rendered_keypoints)
        t4 = time.time()
        performance_accumulator.add_batch_times([t1, t2, t3, t4])

    fecal_boli_results.results_receiver_queue.put((None, None))
    fecal_boli_counts.results_receiver_queue.put((None, None))
    return (fecal_boli_results, fecal_boli_counts, performance_accumulator)


def infer_fecal_boli_pytorch(args):
    """Main function to run a single mouse pose model."""
    model_definition = FECAL_BOLI[args.model]
    cfg.defrost()
    cfg.merge_from_file(model_definition["pytorch-config"])
    cfg.TEST.MODEL_FILE = model_definition["pytorch-model"]
    cfg.freeze()
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    # allow tensor cores
    torch.backends.cuda.matmul.allow_tf32 = True
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
        frame_iter = iter([single_frame])

    fecal_boli_results, fecal_boli_counts, performance_accumulator = predict_fecal_boli(
        frame_iter, model, args.out_video, args.frame_interval, args.batch_size
    )
    final_fecal_boli_detections = fecal_boli_results.get_results()
    final_fecal_boli_counts = fecal_boli_counts.get_results()
    write_fecal_boli_data(
        args.out_file,
        final_fecal_boli_detections,
        final_fecal_boli_counts,
        args.frame_interval,
        model_definition["model-name"],
        model_definition["model-checkpoint"],
    )
    performance_accumulator.print_performance()
