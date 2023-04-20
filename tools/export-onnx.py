import numpy as np
import skimage.transform
import imageio

import torch
import torch.nn.functional as torchfunc
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import sys
sys.path.append('/deep-hres-net/')
sys.path.append('/deep-hres-net/tools/')


import _init_paths
import utils.assocembedutil as aeutil
from config import cfg
from config import update_config

import models

config_file = '/media/bgeuther/Storage/TempStorage/trained-models/2022_pose/multimouse_cloudfactory.yml'
model_file = '/media/bgeuther/Storage/TempStorage/trained-models/2022_pose/best_state.pth'
out_onnx_model = '/media/bgeuther/Storage/TempStorage/trained-models/2022_pose/2022_pose.onnx'

cfg.defrost()
cfg.merge_from_file(config_file)
cfg.TEST.MODEL_FILE = model_file
cfg.freeze()

xform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(
		mean=[0.45, 0.45, 0.45],
		std=[0.225, 0.225, 0.225],
	),
])

cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
	cfg, is_train=False
)

model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
model.eval()
# model = model.cuda()

batch = []
# image = np.zeros([800,800,3], dtype=np.uint8)
test_fname = '/media/bgeuther/Storage/TempStorage/trained-models/2022_pose/test-in.png'
image = imageio.imread(test_fname)
image = xform(image)
batch.append(image)
batch_tensor = torch.stack(batch)

torch.onnx.export(model,
	batch_tensor,
	out_onnx_model,
	export_params=True,
	opset_version=11,
	do_constant_folding=True,
	input_names = ['input'],
	output_names = ['output'],
	dynamic_axes={'input' : {0 : 'batch_size'},
				'output' : {0 : 'batch_size'}})


model_out = model(batch_tensor)
preds = model_out.cpu().detach().numpy()
out_png = '/media/bgeuther/Storage/TempStorage/trained-models/2022_pose/gt-preds.png'
imageio.imwrite(out_png, (np.clip(preds[0,0,:,:]*40,0,1)*255).astype(np.uint8))

tmp = np.clip(preds[0,0,:,:],0,1)
1/np.max(tmp)
