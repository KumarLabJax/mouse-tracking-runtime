"""Inference function for executing ORT for a single mouse pose model."""
import onnx
import onnxruntime
import imageio
import h5py
import numpy as np
import queue
import time
import sys
from utils.pose import argmax_2d, render_pose_overlay
from utils.segmentation import get_frame_masks
from utils.prediction_saver import prediction_saver
from utils.writers import write_pose_v2_data, write_pose_v3_data, adjust_pose_version
from utils.timers import time_accumulator
from models.model_definitions import MULTI_MOUSE_POSE


def infer_multi_pose_ort(args):
	"""Main function to run a single mouse pose model."""
	model_definition = MULTI_MOUSE_POSE[args.model]
	model = onnx.load_model(model_definition['ort-model'])
	onnx.checker.check_model(model)

	options = onnxruntime.SessionOptions()
	options.inter_op_num_threads = 1
	options.intra_op_num_threads = 1
	options.enable_mem_pattern = True
	options.enable_cpu_mem_arena = False
	options.enable_mem_reuse = True
	options.log_severity_level = 1
	options.log_verbosity_level = 1

	ort_session = onnxruntime.InferenceSession(model_definition['ort-model'], providers=[('CUDAExecutionProvider', {'device_id': 0, 'arena_extend_strategy': 'kNextPowerOfTwo', 'gpu_mem_limit': 1 * 1024 * 1024 * 1024, 'cudnn_conv_algo_search': 'EXHAUSTIVE', 'do_copy_in_default_stream': True}), 'CPUExecutionProvider'], sess_options=options)

	if args.video:
		vid_reader = imageio.get_reader(args.video)
		frame_iter = vid_reader.iter_data()
	else:
		single_frame = imageio.imread(args.frame)
		frame_iter = [single_frame]

	pose_results = prediction_saver(dtype=np.uint16)
	confidence_results = prediction_saver(dtype=np.float32)
	vid_writer = None
	if args.out_video is not None:
		vid_writer = imageio.get_writer(args.out_video, fps=30)
	performance_accumulator = time_accumulator(3, ['Preprocess', 'GPU Compute', 'Postprocess'])
	# Main loop for inference
	for frame_idx, frame in enumerate(frame_iter):
		if model_definition['model-name'] == 'topdown':
			t1 = time.time()
			with h5py.File(args.out_file, 'r') as f:
				seg_data = f['poseest/seg_data'][frame_idx]
			masks = get_frame_masks(seg_data, frame.shape[:2])
			frame_batches = []
			for current_mask_idx in range(len(masks)):
				# Skip if no mask
				if not np.any(masks[current_mask_idx]):
					continue
				batch = (np.repeat(255 - masks[current_mask_idx], 3).reshape(frame.shape) + (np.repeat(masks[current_mask_idx], 3).reshape(frame.shape) * frame)).astype(np.float32)
				batch = np.transpose((np.expand_dims(batch, 0) / 255. - 0.45) / 0.225, [0, 3, 1, 2])
				frame_batches.append(batch)
			t2 = time.time()
			predictions = []
			for current_input in frame_batches:
				ort_inputs = {ort_session.get_inputs()[0].name: current_input}
				ort_outs = ort_session.run(None, ort_inputs)
				predictions.append(ort_outs[0])
			t3 = time.time()
			stacked_pose_outputs = np.full([len(predictions), 12, 2], 0, np.int64)
			stacked_conf_outputs = np.full([len(predictions), 12], 0, np.float32)
			for i, current_output in enumerate(predictions):
				predicted_conf, predicted_pose = argmax_2d(current_output)
				stacked_pose_outputs[i] = predicted_pose
				stacked_conf_outputs[i] = predicted_conf
				# TODO: zero data with low conf?
			if vid_writer is not None:
				rendered_pose = np.copy(frame)
				for i in range(len(stacked_pose_outputs)):
					rendered_pose = render_pose_overlay(rendered_pose, stacked_pose_outputs[i], [])
				vid_writer.append_data(rendered_pose)
			try:
				pose_results.results_receiver_queue.put((1, np.expand_dims(stacked_pose_outputs, axis=0)), timeout=5)
				confidence_results.results_receiver_queue.put((1, np.expand_dims(stacked_conf_outputs, 0)), timeout=5)
			except queue.Full:
				if not pose_results.is_healthy() or not confidence_results.is_healthy():
					print('Writer thread died unexpectedly.', file=sys.stderr)
					sys.exit(1)
				print(f'WARNING: Skipping inference on frame {frame_idx}')
				continue
			t4 = time.time()
			performance_accumulator.add_batch_times([t1, t2, t3, t4])
		else:
			raise NotImplementedError('Bottom up model not yet supported.')
			input_frame = np.expand_dims(frame.astype(np.float32), [0])
			input_frame = np.transpose((input_frame / 255. - 0.45) / 0.225, [0, 3, 1, 2])

	pose_results.results_receiver_queue.put((None, None))
	confidence_results.results_receiver_queue.put((None, None))
	pose_matrix = pose_results.get_results()
	confidence_matrix = confidence_results.get_results()
	write_pose_v2_data(args.out_file, pose_matrix, confidence_matrix, model_definition['model-name'], model_definition['model-checkpoint'])
	# Make up fake data for v3 data...
	instance_count = np.sum(np.any(confidence_matrix > 0, axis=2), axis=1).astype(np.uint8)
	instance_embedding = np.full(confidence_matrix.shape, 0, dtype=np.float32)
	# TODO: Make a better dummy (low cost) tracklet generation or allow user to pick one...
	# This one essentially produces valid but horrible data (index means idenitity)
	instance_track_id = np.tile([np.arange(confidence_matrix.shape[1])], confidence_matrix.shape[0]).reshape(confidence_matrix.shape[:2]).astype(np.uint32)
	# instance_track_id = np.zeros(confidence_matrix.shape[:2], dtype=np.uint32)
	for row in range(len(instance_track_id)):
		valid_poses = instance_count[row]
		instance_track_id[row, instance_track_id[row] >= valid_poses] = 0
	write_pose_v3_data(args.out_file, instance_count, instance_embedding, instance_track_id)
	# Since this is topdown, segmentation is present and we can instruct it that it's there
	adjust_pose_version(args.out_file, 6)
	performance_accumulator.print_performance()
