"""Definitions of trained models."""

SINGLE_MOUSE_SEGMENTATION = {
	'tracking-paper': {
		'model-name': 'full-model-tracking-paper',
		'model-checkpoint': 'model.ckpt-415000',
		'ort-model': '/kumar_lab_models/models/ort-models/single-mouse-segmentation/tracking-paper.onnx',
		'tfs-model': '/kumar_lab_models/models/tfs-models/single-mouse-segmentation/tracking-paper/',
	},
}

MULTI_MOUSE_SEGMENTATION = {
	'social-paper': {
		'model-name': 'panoptic-deeplab-res50_v2',
		'model-checkpoint': 'ckpt-125000',
		'ort-model': None,
		'tfs-model': '/kumar_lab_models/models/tfs-models/multi-mouse-segmentation/panoptic-deeplab/',
	},
}

SINGLE_MOUSE_POSE = {
	'gait-paper': {
		'model-name': 'gait-model',
		'model-checkpoint': '2019-06-26-param-search/mp-conf4.yaml',
		'ort-model': '/kumar_lab_models/models/ort-models/single-mouse-pose/gait-model.onnx',
		'tfs-model': None,
	},
}

# Static Objects

STATIC_ARENA_CORNERS = {
	'social-2022-pipeline': {
		'model-name': 'obj-api-kp',
		'model-checkpoint': '2022-11-21/ckpt-101',
		'ort-model': '/kumar_lab_models/models/ort-models/static-objects/obj-api-corners.onnx',
		'tfs-model': '/kumar_lab_models/models/tfs-models/static-object-arena/obj-api-2022/',
	},
}

STATIC_FOOD_CORNERS = {
	'social-2022-pipeline': {
		'model-name': 'obj-api-seg',
		'model-checkpoint': '2022-11-28/ckpt-101',
		'ort-model': '/kumar_lab_models/models/ort-models/static-objects/obj-api-food.onnx',
		'tfs-model': '/kumar_lab_models/models/tfs-models/static-object-food/obj-api-2022/',
	},
}
