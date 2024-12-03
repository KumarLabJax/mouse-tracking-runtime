"""Definitions of trained models."""

SINGLE_MOUSE_SEGMENTATION = {
	'tracking-paper': {
		'model-name': 'full-model-tracking-paper',
		'model-checkpoint': 'model.ckpt-415000',
		'tfs-model': '/kumar_lab_models/models/tfs-models/single-mouse-segmentation/tracking-paper/',
	},
}

MULTI_MOUSE_SEGMENTATION = {
	'social-paper': {
		'model-name': 'panoptic-deeplab-res50_v2',
		'model-checkpoint': 'ckpt-125000',
		'tfs-model': '/kumar_lab_models/models/tfs-models/multi-mouse-segmentation/panoptic-deeplab/',
	},
}

SINGLE_MOUSE_POSE = {
	'gait-paper': {
		'model-name': 'gait-model',
		'model-checkpoint': '2019-06-26-param-search/mp-conf4.yaml',
		'tfs-model': None,
		'lightning-config': '/kumar_lab_models/models/lightning-models/single-mouse-pose/gait-model.yaml',
		'lightning-model': '/kumar_lab_models/models/lightning-models/single-mouse-pose/gait-model.pth',
	},
}

MULTI_MOUSE_POSE = {
	'social-paper-topdown': {
		'model-name': 'topdown',
		'model-checkpoint': 'multimouse_topdown_1.yaml',
		'tfs-model': None,
		'lightning-config': '/kumar_lab_models/models/lightning-models/multi-mouse-pose/social-topdown.yaml',
		'lightning-model': '/kumar_lab_models/models/lightning-models/multi-mouse-pose/social-topdown.pth',
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
		'tfs-model': '/kumar_lab_models/models/tfs-models/multi-mouse-identity/mnas_2021/',
	},
	'2023': {
		'model-name': 'TrackIDTrain_MNAS_latent16',
		'model-checkpoint': 'model.ckpt-290566',
		'tfs-model': '/kumar_lab_models/models/tfs-models/multi-mouse-identity/mnas_2023/',
	}
}

# Fecal Boli

FECAL_BOLI = {
	'fecal-boli': {
		'model-name': 'fecal-boli',
		'model-checkpoint': 'fecalboli/fecalboli_2020-06-19_02.yaml',
		'tfs-model': None,
		'lightning-config': '/kumar_lab_models/models/lightning-models/fecal-boli/fecalboli-2020-06-19.yaml',
		'lightning-model': '/kumar_lab_models/models/lightning-models/fecal-boli/fecalboli-2020-06-19.pth',
	}
}

# Static Objects

STATIC_ARENA_CORNERS = {
	'social-2022-pipeline': {
		'model-name': 'obj-api-kp',
		'model-checkpoint': '2022-11-21/ckpt-101',
		'tfs-model': '/kumar_lab_models/models/tfs-models/static-object-arena/obj-api-2022/',
	},
}

STATIC_FOOD_CORNERS = {
	'social-2022-pipeline': {
		'model-name': 'obj-api-seg',
		'model-checkpoint': '2022-11-28/ckpt-101',
		'tfs-model': '/kumar_lab_models/models/tfs-models/static-object-food/obj-api-2022/',
	},
}

STATIC_LIXIT = {
	'social-2022-pipeline': {
		'model-name': 'dlc-lixit',
		'model-checkpoint': 'iteration-0/final-aug-lixitJan3-trainset95shuffle1/train/snapshot-200000',
		'tfs-model': '/kumar_lab_models/models/tfs-models/static-object-lixit/dlc-2022/',
	},
}
