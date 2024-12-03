import onnx
import onnxruntime
import imageio
import numpy as np
import time
from models.model_definitions import MULTI_MOUSE_POSE
import h5py
from utils.segmentation import get_frame_masks
from utils.pose import argmax_2d


out_onnx_model = MULTI_MOUSE_POSE['social-paper-topdown']
model = onnx.load_model(out_onnx_model['ort-model'])
onnx.checker.check_model(model)

options = onnxruntime.SessionOptions()
options.inter_op_num_threads = 1
options.intra_op_num_threads = 1
options.enable_mem_pattern = True
options.enable_cpu_mem_arena = False
options.enable_mem_reuse = True
options.log_severity_level = 1
options.log_verbosity_level = 1

ort_session = onnxruntime.InferenceSession(out_onnx_model['ort-model'], providers=[('CUDAExecutionProvider', {'device_id': 0, 'arena_extend_strategy': 'kNextPowerOfTwo', 'gpu_mem_limit': 1 * 1024 * 1024 * 1024, 'cudnn_conv_algo_search': 'EXHAUSTIVE', 'do_copy_in_default_stream': True}), 'CPUExecutionProvider'], sess_options=options)

frame = 0
test_frame = imageio.get_reader('../tests/test-multi-vid.mp4').get_data(frame)
with h5py.File('../tests/test-multi-vid_pose_est_v6.h5', 'r') as f:
	seg_data = f['poseest/seg_data'][frame]

masks = get_frame_masks(seg_data, test_frame.shape[:2])
n_valid_masks = np.sum([np.any(x) for x in masks])
for i in range(n_valid_masks):
	batch = (np.repeat(255  -masks[i], 3).reshape(test_frame.shape)+(np.repeat(masks[i], 3).reshape(test_frame.shape)*test_frame)).astype(np.float32)
	batch = np.transpose((np.expand_dims(batch, 0) / 255. - 0.45) / 0.225, [0, 3, 1, 2])
	ort_inputs = {ort_session.get_inputs()[0].name: batch}
	ort_outs = ort_session.run(None, ort_inputs)
	predicted_pose = argmax_2d(ort_outs[0])

deltas = []

for i in np.arange(101):
	start_time = time.time()
	ort_inputs = {ort_session.get_inputs()[0].name: batch}
	start_time = time.time()
	ort_outs = ort_session.run(None, ort_inputs)
	delta = time.time() - start_time
	deltas.append(delta)

print('Last Batch Time: ' + str(delta))
# First batch will take longer, since it runs the optimization
print('Average: ' + str(np.round(np.mean(deltas[1:]), 5)) + ' (' + str(np.round(1 / np.mean(deltas[1:]), 2)) + ' fps)')

