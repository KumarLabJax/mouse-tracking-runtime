"""Inference function for executing TFS for a single mouse segmentation model."""

import queue
import sys
import time

import cv2
import imageio
import numpy as np
import tensorflow.compat.v1 as tf

from mouse_tracking.models.model_definitions import SINGLE_MOUSE_SEGMENTATION
from mouse_tracking.utils.prediction_saver import prediction_saver
from mouse_tracking.utils.segmentation import (
    get_contours,
    pad_contours,
    render_segmentation_overlay,
)
from mouse_tracking.utils.timers import time_accumulator
from mouse_tracking.utils.writers import write_seg_data


def infer_single_segmentation_tfs(args):
    """Main function to run a single mouse segmentation model."""
    model_definition = SINGLE_MOUSE_SEGMENTATION[args.model]
    core_config = tf.ConfigProto()
    core_config.gpu_options.allow_growth = True

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

    with tf.Session(graph=tf.Graph(), config=core_config) as session:
        _model = tf.saved_model.loader.load(
            session, ["serve"], model_definition["tfs-model"]
        )
        graph = tf.get_default_graph()
        input_tensor = graph.get_tensor_by_name("Input_Variables/Placeholder:0")
        output_tensor = graph.get_tensor_by_name("Network/SegmentDecoder/seg/Relu:0")

        # Main loop for inference
        for frame_idx, frame in enumerate(frame_iter):
            t1 = time.time()
            input_frame = np.reshape(
                cv2.resize(frame[:, :, 0], [480, 480]), [1, 480, 480, 1]
            ).astype(np.float32)
            t2 = time.time()
            prediction = session.run(
                [output_tensor], feed_dict={input_tensor: input_frame}
            )
            t3 = time.time()
            predicted_mask = (
                prediction[0][0, :, :, 1] < prediction[0][0, :, :, 0]
            ).astype(np.uint8)
            contours, flags = get_contours(predicted_mask)
            contour_matrix = pad_contours(contours)
            if len(flags) > 0:
                flag_matrix = np.asarray(flags[0][:, 3] == -1).reshape([1, 1, -1])
            else:
                flag_matrix = np.zeros([0])
            try:
                segmentation_results.results_receiver_queue.put(
                    (1, np.expand_dims(contour_matrix, (0, 1))), timeout=500
                )
                seg_flag_results.results_receiver_queue.put(
                    (1, flag_matrix), timeout=500
                )
                if vid_writer is not None:
                    rendered_segmentation = render_segmentation_overlay(
                        contour_matrix, frame
                    )
                    vid_writer.append_data(rendered_segmentation)
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
    )
    performance_accumulator.print_performance()
