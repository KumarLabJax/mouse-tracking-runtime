"""Inference function for executing ORT for a fecal boli model."""
import onnx
import onnxruntime
import imageio
import numpy as np
import cv2
import queue
import time
import sys
import itertools
from utils.pose import localmax_2d
from utils.static_objects import plot_keypoints
from utils.prediction_saver import prediction_saver
from utils.timers import time_accumulator
from models.model_definitions import FECAL_BOLI


def infer_fecal_boli_model(args):
	"""Main function to run a fecal boli model."""
	model_definition = FECAL_BOLI[args.model]
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

	fecal_boli_results = prediction_saver(dtype=np.uint16)
	vid_writer = None
	if args.out_video is not None:
		vid_writer = imageio.get_writer(args.out_video, fps=30)
	performance_accumulator = time_accumulator(3, ['Preprocess', 'GPU Compute', 'Postprocess'])
	# Main loop for inference
	for frame_idx, frame in enumerate(itertools.islice(frame_iter, 0, None, args.frame_interval)):
	# for frame_idx, frame in enumerate(frame_iter):
	# 	if frame_idx % args.frame_interval != 0:
	# 		continue
		t1 = time.time()
		input_frame = np.expand_dims(frame.astype(np.float32), [0])
		input_frame = np.transpose((input_frame / 255. - 0.45) / 0.225, [0, 3, 1, 2])
		ort_inputs = {ort_session.get_inputs()[0].name: input_frame}
		t2 = time.time()
		ort_outs = ort_session.run(['output'], ort_inputs)
		t3 = time.time()
		peaks, locations = localmax_2d(ort_outs[0][0, 0], 0.75, 5)
		# Always write to the video
		if vid_writer is not None:
			render = plot_keypoints(locations, frame, is_yx=True)
			vid_writer.append_data(render)
		try:
			fecal_boli_results.results_receiver_queue.put((1, np.asarray([[len(peaks)]])), timeout=5)
		except queue.Full:
			if not fecal_boli_results.is_healthy():
				print('Writer thread died unexpectedly.', file=sys.stderr)
				sys.exit(1)
			print(f'WARNING: Skipping inference on frame {frame_idx}')
			continue
		t4 = time.time()
		performance_accumulator.add_batch_times([t1, t2, t3, t4])
	fecal_boli_results.results_receiver_queue.put((None, None))
	fecal_boli_counts = fecal_boli_results.get_results()
	print(fecal_boli_counts)
	if args.out_image is not None:
		render = plot_keypoints(locations, frame, is_yx=True)
		imageio.imwrite(args.out_image, render)

	performance_accumulator.print_performance()
