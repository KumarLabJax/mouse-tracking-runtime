import onnx
import onnxruntime
import imageio
import numpy as np
import time

out_onnx_model = 'onnx-models/multi-mouse-pose/2022_pose.onnx'
model = onnx.load_model(out_onnx_model)
onnx.checker.check_model(model)

options = onnxruntime.SessionOptions()
options.inter_op_num_threads = 1
options.intra_op_num_threads = 1
options.enable_mem_pattern = True
options.enable_cpu_mem_arena = False
options.enable_mem_reuse = True
options.log_severity_level = 1
options.log_verbosity_level = 1

ort_session = onnxruntime.InferenceSession(out_onnx_model, providers=[('CUDAExecutionProvider', {'device_id': 0, 'arena_extend_strategy': 'kNextPowerOfTwo', 'gpu_mem_limit': 1 * 1024 * 1024 * 1024, 'cudnn_conv_algo_search': 'EXHAUSTIVE', 'do_copy_in_default_stream': True}), 'CPUExecutionProvider'], sess_options=options)

test_fname = 'tests/test-multi-frame.png'
test_frame = np.expand_dims(imageio.imread(test_fname).astype(np.float32), [0])
test_frame = np.transpose((test_frame / 255. - 0.45) / 0.225, [0, 3, 1, 2])

deltas = []

for i in np.arange(101):
	start_time = time.time()
	ort_inputs = {ort_session.get_inputs()[0].name: test_frame}
	start_time = time.time()
	ort_outs = ort_session.run(None, ort_inputs)
	delta = time.time() - start_time
	deltas.append(delta)

print('Last Batch Time: ' + str(delta))
# First batch will take longer, since it runs the optimization
print('Average: ' + str(np.round(np.mean(deltas[1:]), 5)) + ' (' + str(np.round(1 / np.mean(deltas[1:]), 2)) + ' fps)')

