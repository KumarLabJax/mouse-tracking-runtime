"""Inference function for executing TFS for a static object model."""

import queue
import sys
import time

import imageio
import numpy as np
import tensorflow as tf
from absl import logging

from mouse_tracking.models.model_definitions import STATIC_LIXIT
from mouse_tracking.utils.prediction_saver import prediction_saver
from mouse_tracking.utils.static_objects import plot_keypoints
from mouse_tracking.utils.timers import time_accumulator
from mouse_tracking.utils.writers import write_static_object_data


def infer_lixit_model(args):
    """Main function to run an arena corner static object model."""
    logging.set_verbosity(logging.ERROR)
    model_definition = STATIC_LIXIT[args.model]

    if args.video:
        vid_reader = imageio.get_reader(args.video)
        frame_iter = vid_reader.iter_data()
    else:
        single_frame = imageio.imread(args.frame)
        frame_iter = [single_frame]

    lixit_results = prediction_saver(dtype=np.float32)
    vid_writer = None
    if args.out_video is not None:
        vid_writer = imageio.get_writer(args.out_video, fps=30)
    performance_accumulator = time_accumulator(
        3, ["Preprocess", "GPU Compute", "Postprocess"]
    )

    model = tf.saved_model.load(model_definition["tfs-model"], tags=["serve"])

    # Main loop for inference
    for frame_idx, frame in enumerate(frame_iter):
        if frame_idx > args.num_frames * args.frame_interval:
            break
        if frame_idx % args.frame_interval != 0:
            continue
        t1 = time.time()
        input_frame = tf.convert_to_tensor(frame.astype(np.float32))
        t2 = time.time()
        prediction = model.signatures["serving_default"](input_frame)
        t3 = time.time()
        try:
            prediction_np = prediction["out"].numpy()
            # Only add to the results if it was good quality
            # Threshold >
            good_keypoints = prediction_np[:, 2] > 0.5
            predicted_keypoints = np.reshape(prediction_np[good_keypoints, :2], [-1, 2])
            lixit_results.results_receiver_queue.put(
                (1, np.expand_dims(predicted_keypoints, axis=0)), timeout=5
            )
            # Always write to the video
            if vid_writer is not None:
                render = plot_keypoints(predicted_keypoints, frame, is_yx=True)
                vid_writer.append_data(render)
        except queue.Full:
            if not lixit_results.is_healthy():
                print("Writer thread died unexpectedly.", file=sys.stderr)
                sys.exit(1)
            print(f"WARNING: Skipping inference on frame {frame_idx}")
            continue
        t4 = time.time()
        performance_accumulator.add_batch_times([t1, t2, t3, t4])

    lixit_results.results_receiver_queue.put((None, None))
    lixit_matrix = lixit_results.get_results()
    # TODO: handle un-sorted multiple lixit predictions.
    # For now, we simply take the median of all predictions.
    lixit_matrix = np.ma.array(
        lixit_matrix,
        mask=np.repeat(np.all(lixit_matrix == 0, axis=-1), 2).reshape(
            lixit_matrix.shape
        ),
    ).reshape([-1, 2])
    if np.all(lixit_matrix.mask):
        print("Lixit was not successfully detected.")
    else:
        filtered_keypoints = np.expand_dims(np.ma.median(lixit_matrix, axis=0), axis=0)
        # lixit data is predicted as [y, x] and is written out [y, x]
        if args.out_file is not None:
            write_static_object_data(
                args.out_file,
                filtered_keypoints,
                "lixit",
                model_definition["model-name"],
                model_definition["model-checkpoint"],
            )
        if args.out_image is not None:
            render = plot_keypoints(filtered_keypoints, frame, is_yx=True)
            imageio.imwrite(args.out_image, render)

    performance_accumulator.print_performance()
