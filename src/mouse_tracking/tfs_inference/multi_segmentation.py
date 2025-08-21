"""Inference function for executing TFS for a single mouse segmentation model."""

import queue
import sys
import time

import imageio
import numpy as np
import tensorflow as tf
from absl import logging

from mouse_tracking.models.model_definitions import MULTI_MOUSE_SEGMENTATION
from mouse_tracking.utils.prediction_saver import prediction_saver
from mouse_tracking.utils.segmentation import (
    get_contours,
    merge_multiple_seg_instances,
    pad_contours,
    render_segmentation_overlay,
)
from mouse_tracking.utils.timers import time_accumulator
from mouse_tracking.utils.writers import write_seg_data


def infer_multi_segmentation_tfs(args):
    """Main function to run a multi mouse segmentation model."""
    logging.set_verbosity(logging.ERROR)
    model_definition = MULTI_MOUSE_SEGMENTATION[args.model]

    if args.video:
        vid_reader = imageio.get_reader(args.video)
        frame_iter = vid_reader.iter_data()
    else:
        single_frame = imageio.imread(args.frame)
        frame_iter = [single_frame]

    segmentation_results = prediction_saver(dtype=np.int32, pad_value=-1)
    seg_flag_results = prediction_saver(dtype=bool)
    vid_writer = None
    if args.out_video is not None:
        vid_writer = imageio.get_writer(args.out_video, fps=30)
    performance_accumulator = time_accumulator(
        3, ["Preprocess", "GPU Compute", "Postprocess"]
    )

    model = tf.saved_model.load(model_definition["tfs-model"])

    # Main loop for inference
    for frame_idx, frame in enumerate(frame_iter):
        t1 = time.time()
        input_frame = np.copy(frame)
        t2 = time.time()
        prediction = model(input_frame)
        t3 = time.time()
        frame_contours = []
        instances = np.unique(prediction["panoptic_pred"])
        instances = np.delete(instances, [0])
        # Only look at "mouse" instances
        panopt_pred = prediction["panoptic_pred"].numpy().squeeze(0)
        frame_contours = []
        frame_flags = []
        # instance 1001-2000 are mouse instances in the deeplab2 custom dataset configuration
        for mouse_instance in instances[instances // 1000 == 1]:
            contours, flags = get_contours(panopt_pred == mouse_instance)
            contour_matrix = pad_contours(contours)
            if len(flags) > 0:
                flag_matrix = np.asarray(flags[0][:, 3] == -1).reshape([-1])
            else:
                flag_matrix = np.zeros([0])
            frame_contours.append(contour_matrix)
            frame_flags.append(flag_matrix)
        combined_contour_matrix, combined_flag_matrix = merge_multiple_seg_instances(
            frame_contours, frame_flags
        )

        if vid_writer is not None:
            rendered_segmentation = frame
            for i in range(combined_contour_matrix.shape[0]):
                rendered_segmentation = render_segmentation_overlay(
                    combined_contour_matrix[i], rendered_segmentation
                )
            vid_writer.append_data(rendered_segmentation)
        try:
            segmentation_results.results_receiver_queue.put(
                (1, np.expand_dims(combined_contour_matrix, (0))), timeout=500
            )
            seg_flag_results.results_receiver_queue.put(
                (1, np.expand_dims(combined_flag_matrix, (0))), timeout=500
            )
        except queue.Full:
            if not segmentation_results.is_healthy():
                print("Writer thread died unexpectedly.", file=sys.stderr)
                sys.exit(1)
            print(f"WARNING: Skipping inference on frame {frame_idx}")
            continue
        t4 = time.time()
        performance_accumulator.add_batch_times([t1, t2, t3, t4])

    segmentation_results.results_receiver_queue.put((None, None))
    seg_flag_results.results_receiver_queue.put((None, None))
    segmentation_matrix = segmentation_results.get_results()
    flag_matrix = seg_flag_results.get_results()
    write_seg_data(
        args.out_file,
        segmentation_matrix,
        flag_matrix,
        model_definition["model-name"],
        model_definition["model-checkpoint"],
        True,
    )
    performance_accumulator.print_performance()
