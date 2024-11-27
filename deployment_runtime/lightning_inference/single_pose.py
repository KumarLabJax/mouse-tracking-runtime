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
	model = eval('hrnet_models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
	model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
	model.eval()
	model = model.cuda()

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
		input_frame = torch.tensor(np.transpose((input_frame / 255. - 0.45) / 0.225, [0, 3, 1, 2]))
		t2 = time.time()
		output = model(input_frame.to('cuda')).cpu().detach().numpy()
		t3 = time.time()
		confidence, pose = argmax_2d(output)
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
