import onnx

out_onnx_model = '/media/bgeuther/Storage/TempStorage/trained-models/2022_pose/2022_pose.onnx'
model = onnx.load_model(out_onnx_model)
onnx.checker.check_model(model)

import onnxruntime
import imageio
import numpy as np
import time

ort_session = onnxruntime.InferenceSession(out_onnx_model, providers=['CPUExecutionProvider'])

test_fname = '/media/bgeuther/Storage/TempStorage/trained-models/2022_pose/test-in.png'
test_frame = np.expand_dims(imageio.imread(test_fname).astype(np.float32),[0])
test_frame = np.transpose((test_frame/255.-0.45)/0.225, [0,3,1,2])
# test_frame = np.transpose(test_frame, [0,3,1,2])

start_time = time.time()
ort_inputs = {ort_session.get_inputs()[0].name: test_frame}
ort_outs = ort_session.run(None, ort_inputs)
delta = time.time()-start_time

print('Elapsed time: ' + str(delta))
# CPU: Elapsed time: 0.6579999923706055
out_png = '/media/bgeuther/Storage/TempStorage/trained-models/2022_pose/test-preds.png'
imageio.imwrite(out_png, (np.clip(ort_outs[0][0,0,:,:]*40,0,1)*255).astype(np.uint8))


tmp = np.clip(ort_outs[0][0,0,:,:],0,1)
1/np.max(tmp)
