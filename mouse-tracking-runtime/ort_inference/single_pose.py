"""Inference function for executing ORT for a single mouse pose model."""
import onnx
import onnxruntime
import imageio
import numpy as np
import queue
import time
import sys
from utils.pose import argmax_2d, render_pose_overlay
from utils.prediction_saver import prediction_saver
from utils.writers import write_pose_v2_data
from utils.timers import time_accumulator
from models.model_definitions import SINGLE_MOUSE_POSE


def infer_single_pose_ort(args):
	"""Main function to run a single mouse pose model."""
	model_definition = SINGLE_MOUSE_POSE[args.model]
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
		t1 = time.time()
		input_frame = np.expand_dims(frame.astype(np.float32), [0])
		input_frame = np.transpose((input_frame / 255. - 0.45) / 0.225, [0, 3, 1, 2])
		ort_inputs = {ort_session.get_inputs()[0].name: input_frame}
		t2 = time.time()
		ort_outs = ort_session.run(None, ort_inputs)
		t3 = time.time()
		confidence, pose = argmax_2d(ort_outs[0])
		try:
			pose_results.results_receiver_queue.put((1, np.expand_dims(pose, axis=0)), timeout=5)
			confidence_results.results_receiver_queue.put((1, confidence), timeout=5)
			if vid_writer is not None:
				rendered_pose = render_pose_overlay(frame, pose, [])
				vid_writer.append_data(rendered_pose)
		except queue.Full:
			if not pose_results.is_healthy() or not confidence_results.is_healthy():
				print('Writer thread died unexpectedly.', file=sys.stderr)
				sys.exit(1)
			print(f'WARNING: Skipping inference on frame {frame_idx}')
			continue
		t4 = time.time()
		performance_accumulator.add_batch_times([t1, t2, t3, t4])
	pose_results.results_receiver_queue.put((None, None))
	confidence_results.results_receiver_queue.put((None, None))
	pose_matrix = pose_results.get_results()
	confidence_matrix = confidence_results.get_results()
	write_pose_v2_data(args.out_file, pose_matrix, confidence_matrix, model_definition['model-name'], model_definition['model-checkpoint'])
	performance_accumulator.print_performance()
