"""Functions related to saving data to pose files."""

import h5py
import numpy as np


def write_pose_data(pose_file, pose_matrix, confidence_matrix, config_str: str = '2019-06-26-param-search/mp-conf4.yaml', model_str: str = 'gait-model.onnx'):
	"""Writes pose_v2 data to a file.
	
	Args:
		pose_file: file to write the pose data to
		pose_matrix: pose data of shape [frame, 12, 2] for one animal
		confidence_matrix: confidence data of shape [frame, 12]
		config_str: string defining the configuration of the model used
		model_str: string defining the checkpoint used

	Raises:
		AssertionError if pose and confidence matrices don't have the same number of frames

	TODO:
		If pose file already exists but is for multi-mouse (v3+), we should redirect the call to promote the data to v3+ (single mouse with all multi-mouse fields)
	"""
	assert pose_matrix.shape[0] == confidence_matrix.shape[0]

	with h5py.File(pose_file, 'a') as out_file:
		if 'poseest/points' in out_file:
			del out_file['poseest/points']
		out_file.create_dataset('poseest/points', data=pose_matrix.astype(np.uint16))
		out_file['poseest/points'].attrs['config'] = config_str
		out_file['poseest/points'].attrs['model'] = model_str
		if 'poseest/confidence' in out_file:
			del out_file['poseest/confidence']
		out_file.create_dataset('poseest/confidence', data=confidence_matrix.astype(np.float32))


def write_identity_data(pose_file, embeddings, config_str: str = 'MNAS_latent16', model_str: str = '2022-04-28_model.ckpt-183819'):
	"""Writes identity prediction data to a pose file.

	Args:
		pose_file: file to write the data to
		embeddings: embedding data of shape [frame, n_animals, embed_dim]
		config_str: string defining the configuration of the model used
		model_str: string defining the checkpoint used

	Raises:
		AssertionError if embedding shapes don't match pose in file.
	"""
	with h5py.File(pose_file, 'a') as out_file:
		assert out_file['poseest/points'].shape[:2] == embeddings.shape[:2]
		if 'poseest/identity_embeds' in out_file:
			del out_file['poseest/identity_embeds']
		out_file.create_dataset('poseest/identity_embeds', data=embeddings.astype(np.float32))
		out_file['poseest/identity_embeds'].attrs['config'] = config_str
		out_file['poseest/identity_embeds'].attrs['model'] = model_str


def write_seg_data(pose_file, seg_contours_matrix, seg_external_flags, config_str: str = '', model_str: str = ''):
	"""Writes segmentation data to a pose file.

	Args:
		pose_file: file to write the data to
		seg_contours_matrix: contour data for segmentation of shape [frame, n_animals, n_contours, max_contour_length, 2]
		seg_external_flags: external flags for each contour of shape [frame, n_animals, n_contours]
		config_str: string defining the configuration of the model used
		model_str: string defining the checkpoint used

	Raises:
		AssertionError if shapes don't match
	"""
	assert seg_contours_matrix.shape[:3] == seg_external_flags[:3]

	with h5py.File(pose_file, 'a') as out_file:
		if 'poseest/seg_data' in out_file:
			del out_file['poseest/seg_data']
		out_file.create_dataset('poseest/seg_data', data=seg_contours_matrix, compression="gzip", compression_opts=9)
		out_file['poseest/seg_data'].attrs['config'] = config_str
		out_file['poseest/seg_data'].attrs['model'] = model_str
		if 'poseest/seg_external_flag' in out_file:
			del out_file['poseest/seg_external_flag']
		out_file.create_dataset('poseest/seg_external_flag', data=seg_external_flags, compression="gzip", compression_opts=9)
