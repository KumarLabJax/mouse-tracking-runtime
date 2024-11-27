"""Inference function for executing lightning for a single mouse pose model."""
import imageio
import numpy as np
import queue
import time
import sys
import os
from utils.pose import argmax_2d, render_pose_overlay
from utils.prediction_saver import prediction_saver
from utils.writers import write_pose_v2_data
from utils.timers import time_accumulator
from models.model_definitions import SINGLE_MOUSE_POSE
import torch
import torch.backends.cudnn as cudnn
# Hacky solution to support hrnets relative path of an identically named module
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..'))
import hrnet.lib.models as hrnet_models
from hrnet.lib.config import cfg


def predict_pose(input_iter, model, render: str = None, batch_size: int = 1):
	"""Main function that processes an iterator.

	Args:
		input_iter: an iterator that will produce frame inputs
		model: pytorch lightning loaded model
		render: optional output file for rendering a prediction video
		batch_size: number of frames to predict per-batch

	Returns:
		tuple of (pose_out, conf_out, performance)
		pose_out: output accumulator for keypoint location data
		conf_out: output accumulator for confidence of keypoint data
		performance: timing performance logs
	"""
	pose_results = prediction_saver(dtype=np.uint16)
	confidence_results = prediction_saver(dtype=np.float32)

	if render is not None:
		vid_writer = imageio.get_writer(render, fps=30)

	performance_accumulator = time_accumulator(3, ['Preprocess', 'GPU Compute', 'Postprocess'], frame_per_batch=batch_size)

	# Main loop for inference
	video_done = False
	batch_num = 0
	while not video_done:
		t1 = time.time()
		batch = []
		batch_count = 0
		for _ in np.arange(batch_size):
			try:
				input_frame = np.expand_dims(next(input_iter).astype(np.float32), [0])
				batch.append(input_frame)
				batch_count += 1
			except StopIteration:
				video_done = True
				break
		if batch_count == 0:
			video_done = True
			break
		# concatenate will squeeze batch dim if it is of size 1, so only concat if > 1
		elif batch_count == 1:
			batch = batch[0]
		elif batch_count > 1:
			batch = np.concatenate(batch)
		batch_tensor = torch.tensor(np.transpose((batch / 255. - 0.45) / 0.225, [0, 3, 1, 2]))
		batch_num += 1

		t2 = time.time()
		with torch.no_grad():
			output = model(batch_tensor.to('cuda')).cpu().detach().numpy()
		t3 = time.time()
		confidence, pose = argmax_2d(output)
		try:
			pose_results.results_receiver_queue.put((batch_count, pose), timeout=5)
			confidence_results.results_receiver_queue.put((batch_count, confidence), timeout=5)
		except queue.Full:
			if not pose_results.is_healthy() or not confidence_results.is_healthy():
				print('Writer thread died unexpectedly.', file=sys.stderr)
				sys.exit(1)
			print(f'WARNING: Skipping inference on batch: {batch_num}, frame: {batch_num * batch_size}')
			continue
		if render is not None:
			for idx in np.arange(batch_count):
				rendered_pose = render_pose_overlay(batch[idx].astype(np.uint8), pose[idx], [])
				vid_writer.append_data(rendered_pose)
		t4 = time.time()
		performance_accumulator.add_batch_times([t1, t2, t3, t4])

	pose_results.results_receiver_queue.put((None, None))
	confidence_results.results_receiver_queue.put((None, None))
	return (pose_results, confidence_results, performance_accumulator)


def infer_single_pose_lightning(args):
	"""Main function to run a single mouse pose model."""
	model_definition = SINGLE_MOUSE_POSE[args.model]
	cfg.defrost()
	cfg.merge_from_file(model_definition['lightning-config'])
	cfg.TEST.MODEL_FILE = model_definition['lightning-model']
	cfg.freeze()
	cudnn.benchmark = False
	torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
	torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
	# allow tensor cores
	torch.backends.cuda.matmul.allow_tf32 = True
	model = eval('hrnet_models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
	model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
	model.eval()
	model = model.cuda()

	if args.video:
		vid_reader = imageio.get_reader(args.video)
		frame_iter = vid_reader.iter_data()
	else:
		single_frame = imageio.imread(args.frame)
		frame_iter = iter([single_frame])

	pose_results, confidence_results, performance_accumulator = predict_pose(frame_iter, model, args.out_video, args.batch_size)
	pose_matrix = pose_results.get_results()
	confidence_matrix = confidence_results.get_results()
	write_pose_v2_data(args.out_file, pose_matrix, confidence_matrix, model_definition['model-name'], model_definition['model-checkpoint'])
	performance_accumulator.print_performance()
