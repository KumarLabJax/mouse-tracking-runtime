"""Inference function for executing TFS for a static object model."""

import queue
import sys
import time

import cv2
import imageio
import numpy as np
import tensorflow.compat.v1 as tf

from mouse_tracking.models.model_definitions import STATIC_ARENA_CORNERS
from mouse_tracking.utils.prediction_saver import prediction_saver
from mouse_tracking.utils.static_objects import (
    ARENA_IMAGING_RESOLUTION,
    DEFAULT_CM_PER_PX,
    filter_square_keypoints,
    get_px_per_cm,
    plot_keypoints,
)
from mouse_tracking.utils.timers import time_accumulator
from mouse_tracking.utils.writers import (
    write_pixel_per_cm_attr,
    write_static_object_data,
)


def infer_arena_corner_model(args):
    """Main function to run an arena corner static object model."""
    model_definition = STATIC_ARENA_CORNERS[args.model]
    core_config = tf.ConfigProto()
    core_config.gpu_options.allow_growth = True

    if args.video:
        vid_reader = imageio.get_reader(args.video)
        frame_iter = vid_reader.iter_data()
    else:
        single_frame = imageio.imread(args.frame)
        frame_iter = [single_frame]

    corner_results = prediction_saver(dtype=np.float32)
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
        det_score = graph.get_tensor_by_name("StatefulPartitionedCall:6")
        # det_class = graph.get_tensor_by_name("StatefulPartitionedCall:2")
        # det_boxes = graph.get_tensor_by_name("StatefulPartitionedCall:0")
        # det_numbs = graph.get_tensor_by_name("StatefulPartitionedCall:7")
        det_keypoint = graph.get_tensor_by_name("StatefulPartitionedCall:4")
        # det_keypoint_score = graph.get_tensor_by_name("StatefulPartitionedCall:3")

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
            scores, keypoints = session.run(
                [det_score, det_keypoint], feed_dict={input_tensor: frame_scaled}
            )
            t3 = time.time()
            try:
                # Keypoints are predicted as [y, x] scaled from 0-1 based on image size
                # Convert to [x, y] pixel units
                predicted_keypoints = np.flip(keypoints[0][0], axis=-1) * np.max(
                    frame.shape
                )
                # Only add to the results if it was good quality
                if scores[0][0] > 0.5:
                    corner_results.results_receiver_queue.put(
                        (1, np.expand_dims(predicted_keypoints, axis=0)), timeout=5
                    )
                # Always write to the video
                if vid_writer is not None:
                    render = plot_keypoints(predicted_keypoints, frame)
                    vid_writer.append_data(render)
            except queue.Full:
                if not corner_results.is_healthy():
                    print("Writer thread died unexpectedly.", file=sys.stderr)
                    sys.exit(1)
                print(f"WARNING: Skipping inference on frame {frame_idx}")
                continue
            t4 = time.time()
            performance_accumulator.add_batch_times([t1, t2, t3, t4])

    corner_results.results_receiver_queue.put((None, None))
    corner_matrix = corner_results.get_results()
    try:
        if corner_matrix is None:
            raise ValueError("No corner predictions were generated")
        filtered_corners = filter_square_keypoints(corner_matrix)
        if args.out_file is not None:
            write_static_object_data(
                args.out_file,
                filtered_corners,
                "corners",
                model_definition["model-name"],
                model_definition["model-checkpoint"],
            )
        px_per_cm = get_px_per_cm(filtered_corners)
        write_pixel_per_cm_attr(args.out_file, px_per_cm, "corner_detection")
        if args.out_image is not None:
            render = plot_keypoints(filtered_corners, frame)
            imageio.imwrite(args.out_image, render)
    except ValueError:
        if frame.shape[0] in ARENA_IMAGING_RESOLUTION:
            print("Corners not successfully detected, writing default px per cm...")
            px_per_cm = DEFAULT_CM_PER_PX[ARENA_IMAGING_RESOLUTION[frame.shape[0]]]
            if args.out_file is not None:
                write_pixel_per_cm_attr(args.out_file, px_per_cm, "default_alignment")
        else:
            print(
                "Corners not successfully detected, arena size not correctly detected from imaging size..."
            )

    performance_accumulator.print_performance()
