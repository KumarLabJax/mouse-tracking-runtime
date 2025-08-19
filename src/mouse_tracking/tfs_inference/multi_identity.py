"""Inference function for executing TFS for a multi-mouse identity model."""

import queue
import sys
import time

import h5py
import imageio
import numpy as np
import tensorflow as tf
from absl import logging

from mouse_tracking.models.model_definitions import MULTI_MOUSE_IDENTITY
from mouse_tracking.utils.identity import (
    InvalidIdentityException,
    crop_and_rotate_frame,
)
from mouse_tracking.utils.prediction_saver import prediction_saver
from mouse_tracking.utils.timers import time_accumulator
from mouse_tracking.utils.writers import write_identity_data


def infer_multi_identity_tfs(args):
    """Main function to run a multi mouse segmentation model."""
    logging.set_verbosity(logging.ERROR)
    model_definition = MULTI_MOUSE_IDENTITY[args.model]

    if args.video:
        vid_reader = imageio.get_reader(args.video)
        frame_iter = vid_reader.iter_data()
    else:
        single_frame = imageio.imread(args.frame)
        frame_iter = [single_frame]

    embedding_results = prediction_saver(dtype=np.float32, pad_value=0)
    performance_accumulator = time_accumulator(
        3, ["Preprocess", "GPU Compute", "Postprocess"]
    )

    with h5py.File(args.out_file, "r") as f:
        pose_data = f["poseest/points"][:]

    model = tf.saved_model.load(model_definition["tfs-model"])
    embed_size = model.signatures["serving_default"].output_shapes["out"][1]

    # Main loop for inference
    for frame_idx, frame in enumerate(frame_iter):
        t1 = time.time()
        input_frames = np.zeros([pose_data.shape[1], 128, 128], dtype=np.uint8)
        valid_poses = np.arange(pose_data.shape[1])
        # Rotate and crop each pose instance
        for animal_idx in np.arange(pose_data.shape[1]):
            try:
                transformed_frame = crop_and_rotate_frame(
                    frame, pose_data[frame_idx, animal_idx], [128, 128]
                )
                input_frames[animal_idx] = transformed_frame[:, :, 0]
            except InvalidIdentityException:
                valid_poses = valid_poses[valid_poses != animal_idx]
        t2 = time.time()
        raw_predictions = []
        for animal_idx in valid_poses:
            prediction = model.signatures["serving_default"](
                tf.convert_to_tensor(input_frames[animal_idx].reshape([1, 128, 128, 1]))
            )
            raw_predictions.append(prediction["out"])
        t3 = time.time()
        prediction_matrix = np.zeros([pose_data.shape[1], embed_size], dtype=np.float32)
        for animal_idx, cur_prediction in zip(
            valid_poses, raw_predictions, strict=False
        ):
            prediction_matrix[animal_idx] = cur_prediction

        try:
            embedding_results.results_receiver_queue.put(
                (1, np.expand_dims(prediction_matrix, (0))), timeout=5
            )
        except queue.Full:
            if not embedding_results.is_healthy():
                print("Writer thread died unexpectedly.", file=sys.stderr)
                sys.exit(1)
            print(f"WARNING: Skipping inference on frame {frame_idx}")
            continue
        t4 = time.time()
        performance_accumulator.add_batch_times([t1, t2, t3, t4])

    embedding_results.results_receiver_queue.put((None, None))
    final_embedding_matrix = embedding_results.get_results()
    write_identity_data(
        args.out_file,
        final_embedding_matrix,
        model_definition["model-name"],
        model_definition["model-checkpoint"],
    )
    performance_accumulator.print_performance()
