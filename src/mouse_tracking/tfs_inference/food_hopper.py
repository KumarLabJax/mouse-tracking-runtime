"""Inference function for executing TFS for a static object model."""

import queue
import sys
import time

import cv2
import imageio
import numpy as np
import tensorflow.compat.v1 as tf

from mouse_tracking.models.model_definitions import STATIC_FOOD_CORNERS
from mouse_tracking.utils.prediction_saver import prediction_saver
from mouse_tracking.utils.static_objects import (
    filter_static_keypoints,
    get_mask_corners,
    plot_keypoints,
)
from mouse_tracking.utils.timers import time_accumulator
from mouse_tracking.utils.writers import write_static_object_data


def infer_food_hopper_model(args):
    """Main function to run an arena corner static object model."""
    model_definition = STATIC_FOOD_CORNERS[args.model]
    core_config = tf.ConfigProto()
    core_config.gpu_options.allow_growth = True

    if args.video:
        vid_reader = imageio.get_reader(args.video)
        frame_iter = vid_reader.iter_data()
    else:
        single_frame = imageio.imread(args.frame)
        frame_iter = [single_frame]

    food_hopper_results = prediction_saver(dtype=np.float32)
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
        input_tensor = graph.get_tensor_by_name("serving_default_input_tensor:0")
        det_score = graph.get_tensor_by_name("StatefulPartitionedCall:5")
        # det_class = graph.get_tensor_by_name("StatefulPartitionedCall:2")
        det_boxes = graph.get_tensor_by_name("StatefulPartitionedCall:0")
        # det_numbs = graph.get_tensor_by_name("StatefulPartitionedCall:6")
        det_mask = graph.get_tensor_by_name("StatefulPartitionedCall:3")

        # Main loop for inference
        for frame_idx, frame in enumerate(frame_iter):
            if frame_idx > args.num_frames * args.frame_interval:
                break
            if frame_idx % args.frame_interval != 0:
                continue
            t1 = time.time()
            frame_scaled = np.expand_dims(
                cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA), axis=0
            )
            t2 = time.time()
            scores, boxes, masks = session.run(
                [det_score, det_boxes, det_mask], feed_dict={input_tensor: frame_scaled}
            )
            t3 = time.time()
            try:
                # Return value is sorted [y1, x1, y2, x2]. Change it to [x1, y1, x2, y2]
                prediction_box = boxes[0][0][[1, 0, 3, 2]]
                # Only add to the results if it was good quality
                predicted_keypoints = get_mask_corners(
                    prediction_box, masks[0][0], frame.shape[:2]
                )
                if scores[0][0] > 0.5:
                    food_hopper_results.results_receiver_queue.put(
                        (1, np.expand_dims(predicted_keypoints, axis=0)), timeout=5
                    )
                # Always write to the video
                if vid_writer is not None:
                    render = plot_keypoints(predicted_keypoints, frame)
                    vid_writer.append_data(render)
            except queue.Full:
                if not food_hopper_results.is_healthy():
                    print("Writer thread died unexpectedly.", file=sys.stderr)
                    sys.exit(1)
                print(f"WARNING: Skipping inference on frame {frame_idx}")
                continue
            t4 = time.time()
            performance_accumulator.add_batch_times([t1, t2, t3, t4])

    food_hopper_results.results_receiver_queue.put((None, None))
    food_hopper_matrix = food_hopper_results.get_results()
    try:
        filtered_keypoints = filter_static_keypoints(food_hopper_matrix)
        # food hopper data is written out [y, x]
        filtered_keypoints = np.flip(filtered_keypoints, axis=-1)
        if args.out_file is not None:
            write_static_object_data(
                args.out_file,
                filtered_keypoints,
                "food_hopper",
                model_definition["model-name"],
                model_definition["model-checkpoint"],
            )
        if args.out_image is not None:
            render = plot_keypoints(filtered_keypoints, frame, is_yx=True)
            imageio.imwrite(args.out_image, render)
    except ValueError:
        print("Food Hopper Corners not successfully detected.")

    performance_accumulator.print_performance()
