import onnx

# out_onnx_model = '/media/bgeuther/Storage/TempStorage/trained-models/2022_pose/2022_pose.onnx'
out_onnx_model = '/projects/kumar-lab/bgeuther/onnx/2022_pose.onnx'
model = onnx.load_model(out_onnx_model)
onnx.checker.check_model(model)

import onnxruntime
import imageio
import numpy as np
import time
import os

options = onnxruntime.SessionOptions()
options.inter_op_num_threads = 1; options.intra_op_num_threads = 1;

ort_session = onnxruntime.InferenceSession(out_onnx_model, providers=['TensorrtExecutionProvider', ('CUDAExecutionProvider',{'device_id': 0, 'arena_extend_strategy': 'kNextPowerOfTwo', 'gpu_mem_limit': 2 * 1024 * 1024 * 1024, 'cudnn_conv_algo_search': 'EXHAUSTIVE', 'do_copy_in_default_stream': True}), 'CPUExecutionProvider'], sess_options=options)

# test_fname = '/media/bgeuther/Storage/TempStorage/trained-models/2022_pose/test-in.png'
test_fname = '/projects/kumar-lab/bgeuther/onnx/test-in.png'
test_frame = np.expand_dims(imageio.imread(test_fname).astype(np.float32),[0])
test_frame = np.transpose((test_frame/255.-0.45)/0.225, [0,3,1,2])
# test_frame = np.transpose(test_frame, [0,3,1,2])

start_time = time.time()
ort_inputs = {ort_session.get_inputs()[0].name: test_frame}
start_time = time.time()
ort_outs = ort_session.run(None, ort_inputs)
delta = time.time()-start_time

print('Elapsed time: ' + str(delta))
# CPU: Elapsed time: 0.6579999923706055
# GPU A100: Elapsed time: 0.03422999382019043
# out_png = '/media/bgeuther/Storage/TempStorage/trained-models/2022_pose/test-preds.png'
out_png = '/projects/kumar-lab/bgeuther/onnx/test-preds.png'
imageio.imwrite(out_png, (np.clip(ort_outs[0][0,0,:,:]*40,0,1)*255).astype(np.uint8))


tmp = np.clip(ort_outs[0][0,0,:,:],0,1)
1/np.max(tmp)
