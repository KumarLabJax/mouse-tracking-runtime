"""Definitions of trained models."""
model_folder = '/kumar_lab_models/models/'
# model_folder = '/media/bgeuther/Storage/TempStorage/onnx/onnx-pipelines/models/'

SINGLE_MOUSE_SEGMENTATION = {
	'tracking-paper': {
		'model-name': 'full-model-tracking-paper',
		'model-checkpoint': 'model.ckpt-415000',
		'tfs-model': model_folder + 'tfs-models/single-mouse-segmentation/tracking-paper/',
	},
}

MULTI_MOUSE_SEGMENTATION = {
	'social-paper': {
		'model-name': 'panoptic-deeplab-res50_v2',
		'model-checkpoint': 'ckpt-125000',
		'tfs-model': model_folder + 'tfs-models/multi-mouse-segmentation/panoptic-deeplab/',
	},
}

SINGLE_MOUSE_POSE = {
	'gait-paper': {
		'model-name': 'gait-model',
		'model-checkpoint': '2019-06-26-param-search/mp-conf4.yaml',
		'tfs-model': None,
		'pytorch-config': model_folder + 'pytorch-models/single-mouse-pose/gait-model.yaml',
		'pytorch-model': model_folder + 'pytorch-models/single-mouse-pose/gait-model.pth',
	},
}

MULTI_MOUSE_POSE = {
	'social-paper-topdown': {
		'model-name': 'topdown',
		'model-checkpoint': 'multimouse_topdown_1.yaml',
		'tfs-model': None,
		'pytorch-config': model_folder + 'pytorch-models/multi-mouse-pose/social-topdown.yaml',
		'pytorch-model': model_folder + 'pytorch-models/multi-mouse-pose/social-topdown.pth',
	},
	'social-paper-bottomup': {
		'model-name': 'bottomup',
		'model-checkpoint': 'multimouse_cloudfactory.yaml',
		'tfs-model': None,
	},
}

MULTI_MOUSE_IDENTITY = {
	'social-paper': {
		'model-name': 'TrackIDTrain_MNAS_latent16',
		'model-checkpoint': 'model.ckpt-183819',
		'tfs-model': model_folder + 'tfs-models/multi-mouse-identity/mnas_2021/',
	},
	'2023': {
		'model-name': 'TrackIDTrain_MNAS_latent16',
		'model-checkpoint': 'model.ckpt-290566',
		'tfs-model': model_folder + 'tfs-models/multi-mouse-identity/mnas_2023/',
	}
}

# Fecal Boli

FECAL_BOLI = {
	'fecal-boli': {
		'model-name': 'fecal-boli',
		'model-checkpoint': 'fecalboli/fecalboli_2020-06-19_02.yaml',
		'tfs-model': None,
		'pytorch-config': model_folder + 'pytorch-models/fecal-boli/fecalboli-2020-06-19.yaml',
		'pytorch-model': model_folder + 'pytorch-models/fecal-boli/fecalboli-2020-06-19.pth',
	}
}

# Static Objects

STATIC_ARENA_CORNERS = {
	'social-2022-pipeline': {
		'model-name': 'obj-api-kp',
		'model-checkpoint': '2022-11-21/ckpt-101',
		'tfs-model': model_folder + 'tfs-models/static-object-arena/obj-api-2022/',
	},
}

STATIC_FOOD_CORNERS = {
	'social-2022-pipeline': {
		'model-name': 'obj-api-seg',
		'model-checkpoint': '2022-11-28/ckpt-101',
		'tfs-model': model_folder + 'tfs-models/static-object-food/obj-api-2022/',
	},
}

STATIC_LIXIT = {
	'social-2022-pipeline': {
		'model-name': 'dlc-lixit',
		'model-checkpoint': 'iteration-0/final-aug-lixitJan3-trainset95shuffle1/train/snapshot-200000',
		'tfs-model': model_folder + 'tfs-models/static-object-lixit/dlc-2022/',
	},
}
