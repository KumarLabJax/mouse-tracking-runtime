"""Inference script for arena corners."""

import argparse
import sys
import os
import onnx
import onnxruntime
import imageio
import numpy as np
import cv2
import queue
import time
from utils.static_objects import filter_square_keypoints, plot_keypoints
from utils.prediction_saver import prediction_saver
from utils.writers import write_static_object_data
from utils.timers import time_accumulator


def infer_corner_model(args):
	"""Main function to run a single mouse pose model."""
	model = onnx.load_model(args.model)
	onnx.checker.check_model(model)

	options = onnxruntime.SessionOptions()
	options.inter_op_num_threads = 1
	options.intra_op_num_threads = 1
	options.enable_mem_pattern = True
	options.enable_cpu_mem_arena = False
	options.enable_mem_reuse = True
	options.log_severity_level = 1
	options.log_verbosity_level = 1

	ort_session = onnxruntime.InferenceSession(args.model, providers=[('CUDAExecutionProvider', {'device_id': 0, 'arena_extend_strategy': 'kNextPowerOfTwo', 'gpu_mem_limit': 1 * 1024 * 1024 * 1024, 'cudnn_conv_algo_search': 'EXHAUSTIVE', 'do_copy_in_default_stream': True}), 'CPUExecutionProvider'], sess_options=options)

	output_names = [x.name for x in ort_session.get_outputs()]

	# Check that this has the outputs we want
	assert 'detection_keypoints' in output_names
	assert 'detection_scores' in output_names

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
	performance_accumulator = time_accumulator(3, ['Preprocess', 'GPU Compute', 'Postprocess'])
	# Main loop for inference
	for frame_idx, frame in enumerate(frame_iter):
		if frame_idx > args.num_frames * args.frame_interval:
			break
		if frame_idx % args.frame_interval != 0:
			continue
		t1 = time.time()
		frame_scaled = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
		frame_scaled = np.expand_dims(frame_scaled.astype(np.uint8), [0])
		ort_inputs = {ort_session.get_inputs()[0].name: frame_scaled}
		t2 = time.time()
		keypoints, scores = ort_session.run(['detection_keypoints', 'detection_scores'], ort_inputs)
		t3 = time.time()
		try:
			# Only add to the results if it was good quality
			if scores[0][0] > 0.5:
				corner_results.results_receiver_queue.put((1, np.expand_dims(keypoints[0][0] * np.max(frame.shape), axis=0)), timeout=5)
			# Always write to the video
			if vid_writer is not None:
				render = plot_keypoints(keypoints[0][0] * np.max(frame.shape), frame)
				vid_writer.append_data(render)
		except queue.Full:
			if not corner_results.is_healthy():
				print('Writer thread died unexpectedly.', file=sys.stderr)
				sys.exit(1)
			print(f'WARNING: Skipping inference on frame {frame_idx}')
			continue
		t4 = time.time()
		performance_accumulator.add_batch_times([t1, t2, t3, t4])
	corner_results.results_receiver_queue.put((None, None))
	corner_matrix = corner_results.get_results()
	try:
		filtered_corners = filter_square_keypoints(corner_matrix)
		write_static_object_data(args.out_file, filtered_corners, 'corners', )
		if args.out_image is not None:
			render = plot_keypoints(filtered_corners, frame)
			imageio.imwrite(args.out_image, render)
	except ValueError:
		print('Corners not successfully detected...')

	performance_accumulator.print_performance()


def main(argv):
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(description='Script that infers an onnx single mouse pose model.')
	parser.add_argument('--model', help='Onnx saved model (*.onnx).', default='/onnx-models/static-objects/obj-api-corners.onnx')
	vid_or_img = parser.add_mutually_exclusive_group(required=True)
	vid_or_img.add_argument('--video', help='Video file for processing')
	vid_or_img.add_argument('--frame', help='Image file for processing')
	parser.add_argument('--out-file', help='Pose file to write out.', required=True)
	parser.add_argument('--out-image', help='Render the final prediction to an image.', default=None)
	parser.add_argument('--out-video', help='Render all predictions to a video.', default=None)
	parser.add_argument('--num-frames', help='Number of frames to predict on.', default=100, type=int)
	parser.add_argument('--frame-interval', help='Interval of frames to predict on.', default=100, type=int)
	#
	args = parser.parse_args()
	if args.video:
		assert os.path.exists(args.video)
	else:
		assert os.path.exists(args.frame)
	infer_corner_model(args)


if __name__ == '__main__':
	main(sys.argv[1:])
