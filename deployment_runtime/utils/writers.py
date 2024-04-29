"""Functions related to saving data to pose files."""

import h5py
import numpy as np
from .matching import hungarian_match_points_seg
from .pose import convert_v2_to_v3


class InvalidPoseFileException(Exception):
	"""Exception if pose data doesn't make sense."""
	def __init__(self, message):   
		"""Just a basic exception with a message."""         
		super().__init__(message)


def promote_pose_data(pose_file, current_version: int, new_version: int):
	"""Promotes the data contained within a pose file to a higher version.

	Args:
		pose_file: pose file containing single mouse pose data to promote
		current_version: current version of the data
		new_version: version to promote the data

	Notes:
		v2 -> v3 changes shape of data from single mouse to multi-mouse
			'poseest/points' from [frame, 12, 2] to [frame, 1, 12, 2]
			'poseest/confidence' from [frame, 12] to [frame, 1, 12]
			'poseest/instance_count', 'poseest/instance_embedding', and 'poseest/instance_track_id' added
		v3 -> v4
			'poseest/id_mask', 'poseest/identity_embeds', 'poseest/instance_embed_id', 'poseest/instance_id_center' added
			This approach will only preserve the longest tracks and does not do any complex stitching
		v4 -> v5
			no change (all data optional)
		v5 -> v6
			'poseest/instance_seg_id' and 'poseest/longterm_seg_id' are assigned to match existing pose data
	"""
	# Promote single mouse data to multimouse
	if current_version < 3 and new_version >= 3:
		with h5py.File(pose_file, 'r') as f:
			pose_data = np.reshape(f['poseest/points'][:], [-1, 1, 12, 2])
			conf_data = np.reshape(f['poseest/confidence'][:], [-1, 1, 12])
			config_str = f['poseest/points'].attrs['config']
			model_str = f['poseest/points'].attrs['model']
		pose_data, conf_data, instance_count, instance_embedding, instance_track_id = convert_v2_to_v3(pose_data, conf_data)
		# Overwrite the existing data with a new axis
		write_pose_v2_data(pose_file, pose_data, conf_data, config_str, model_str)
		write_pose_v3_data(pose_file, instance_count, instance_embedding, instance_track_id)
		current_version = 3

	# Add in v4 fields
	if current_version < 4 and new_version >= 4:
		with h5py.File(pose_file, 'r') as f:
			track_data = f['poseest/instance_track_id'][:]
			instance_data = f['poseest/instance_count'][:]
		# Preserve longest tracks
		num_mice = np.max(instance_data)
		mouse_idxs = np.repeat([np.arange(track_data.shape[1])], track_data.shape[0], axis=0)
		valid_idxs = np.repeat(np.reshape(instance_data, [-1, 1]), track_data.shape[1], axis=1)
		masked_track_data = np.ma.array(track_data, mask=mouse_idxs > valid_idxs)
		tracks, track_frame_counts = np.unique(masked_track_data, return_counts=True)
		# Generate dummy data
		masks = np.full(track_data.shape, True, dtype=bool)
		embeds = np.full([track_data.shape[0], 1], 0, dtype=np.float32)
		ids = np.full(track_data.shape, 0, dtype=np.uint32)
		centers = np.full([1, num_mice], 0, dtype=np.float64)
		# Special case where we can just flatten all tracklets into 1 id
		if num_mice == 1:
			for cur_track in tracks:
				observations = track_data == cur_track
				masks[observations] = False
				ids[observations] = 1
		# Non-trivial case where we simply select the longest tracks and keep them.
		# We could potentially try and stitch tracklets, but that should be explicit.
		else:
			tracks_to_keep = tracks[np.argsort(track_frame_counts)[:num_mice]]
			for i, cur_track in enumerate(tracks_to_keep):
				observations = track_data == cur_track
				masks[observations] = False
				ids[observations] = i + 1
		write_pose_v4_data(pose_file, masks, ids, centers, embeds)
		current_version = 4

	# Match segmentation data with pose data
	if current_version < 6 and new_version >= 6:
		with h5py.File(pose_file, 'r') as f:
			pose_data = f['poseest/points'][:]
			pose_tracks = f['poseest/instance_track_id'][:]
			pose_ids = f['poseest/instance_embed_id'][:]
			seg_data = f['poseest/seg_data'][:]
		seg_tracks = np.full(pose_tracks.shape, 0, dtype=np.uint32)
		seg_ids = np.full(pose_tracks.shape, 0, dtype=np.uint32)
		for frame in np.arange(pose_tracks.shape[0]):
			matches = hungarian_match_points_seg(pose_data[frame], seg_data[frame])
			for current_match in matches:
				seg_tracks[frame, current_match[1]] = pose_tracks[frame, current_match[0]]
				seg_ids[frame, current_match[1]] = pose_ids[frame, current_match[0]]
		write_v6_tracklets(pose_file, seg_tracks, seg_ids)
		current_version = 6


def adjust_pose_version(pose_file, version: int):
	"""Safely adjusts the pose version.

	Args:
		pose_file: file to change the stored pose version
		version: new version to use

	Raises:
		ValueError if version is not within a valid range
	"""
	if version < 2 or version > 6:
		raise ValueError(f'Pose version {version} not allowed. Please select between 2-6.')

	with h5py.File(pose_file, 'r') as in_file:
		try:
			current_version = in_file['poseest'].attrs['version'][0]
		# KeyError can be either group or version not being present
		# IndexError would be incorrect shape of the version attribute
		except (KeyError, IndexError):
			if 'poseest' not in in_file:
				in_file.create_group('poseest')
			current_version = -1
	if current_version < version:
		# Change the value before promoting data.
		# `promote_pose_data` will call this function again, but will skip this because the version has already been promoted
		with h5py.File(pose_file, 'a') as out_file:
			out_file['poseest'].attrs['version'] = np.asarray([version, 0], dtype=np.uint16)
		promote_pose_data(pose_file, current_version, version)


def write_pose_v2_data(pose_file, pose_matrix: np.ndarray, confidence_matrix: np.ndarray, config_str: str = '', model_str: str = ''):
	"""Writes pose_v2 data fields to a file.
	
	Args:
		pose_file: file to write the pose data to
		pose_matrix: pose data of shape [frame, 12, 2] for one animal and [frame, num_animals, 12, 2] for multi-animal
		confidence_matrix: confidence data of shape [frame, 12] for one animal and [frame, num_animals, 12] for multi-animal
		config_str: string defining the configuration of the model used
		model_str: string defining the checkpoint used

	Raises:
		InvalidPoseFileException if pose and confidence matrices don't have the same number of frames
	"""
	if pose_matrix.shape[0] != confidence_matrix.shape[0]:
		raise InvalidPoseFileException('Pose data does not match confidence data.')

	with h5py.File(pose_file, 'a') as out_file:
		if 'poseest/points' in out_file:
			del out_file['poseest/points']
		out_file.create_dataset('poseest/points', data=pose_matrix.astype(np.uint16))
		out_file['poseest/points'].attrs['config'] = config_str
		out_file['poseest/points'].attrs['model'] = model_str
		if 'poseest/confidence' in out_file:
			del out_file['poseest/confidence']
		out_file.create_dataset('poseest/confidence', data=confidence_matrix.astype(np.float32))

	adjust_pose_version(pose_file, 2)


def write_pose_v3_data(pose_file, instance_count: np.ndarray = None, instance_embedding: np.ndarray = None, instance_track: np.ndarray = None):
	"""Writes pose_v3 data fields to a file.

	Args:
		pose_file: file to write the pose data to
		instance_count: count of valid instances per frame of shape [frame]
		instance_embedding: associative embedding values for keypoints of shape [frame, num_animals, 12]
		instance_track: track id for the tracklet data of shape [frame, num_animals]

	Raises:
		InvalidPoseFileException if a required dataset was either not provided or not present in the file
	"""
	with h5py.File(pose_file, 'a') as out_file:
		if instance_count is not None:
			if 'poseest/instance_count' in out_file:
				del out_file['poseest/instance_count']
			out_file.create_dataset('poseest/instance_count', data=instance_count.astype(np.uint8))
		else:
			if 'poseest/instance_count' not in out_file:
				raise InvalidPoseFileException('Instance count field was not provided and is required.')
		if instance_embedding is not None:
			if 'poseest/instance_embedding' in out_file:
				del out_file['poseest/instance_embedding']
			out_file.create_dataset('poseest/instance_embedding', data=instance_embedding.astype(np.float32))
		else:
			if 'poseest/instance_embedding' not in out_file:
				raise InvalidPoseFileException('Instance embedding field was not provided and is required.')
		if instance_track is not None:
			if 'poseest/instance_track_id' in out_file:
				del out_file['poseest/instance_track_id']
			out_file.create_dataset('poseest/instance_track_id', data=instance_track.astype(np.uint32))
		else:
			if 'poseest/instance_track_id' not in out_file:
				raise InvalidPoseFileException('Instance track id field was not provided and is required.')

	adjust_pose_version(pose_file, 3)


def write_pose_v4_data(pose_file, mask: np.ndarray, longterm_ids: np.ndarray, centers: np.ndarray, embeddings: np.ndarray = None):
	"""Writes pose_v4 data fields to a file.

	Args:
		pose_file: file to write the pose data to
		mask: identity masking data (0 = visible data, 1 = masked data) of shape [frame, num_animals]
		longterm_ids: longterm identity assignments of shape [frame, num_animals]
		centers: embedding centers of shape [num_ids, embed_dim]
		embeddings: identity embedding vectors of shape [frame, num_animals, embed_dim]

	Raises:
		InvalidPoseFileException if a required dataset was either not provided or not present in the file
	"""
	with h5py.File(pose_file, 'a') as out_file:
		if 'poseest/id_mask' in out_file:
			del out_file['poseest/id_mask']
		out_file.create_dataset('poseest/id_mask', data=mask.astype(bool))
		if 'poseest/instance_embed_id' in out_file:
			del out_file['poseest/instance_embed_id']
		out_file.create_dataset('poseest/instance_embed_id', data=longterm_ids.astype(np.uint32))
		if 'poseest/instance_id_center' in out_file:
			del out_file['poseest/instance_id_center']
		out_file.create_dataset('poseest/instance_id_center', data=centers.astype(np.float64))
		if embeddings is not None:
			if 'poseest/identity_embeds' in out_file:
				del out_file['poseest/identity_embeds']
			out_file.create_dataset('poseest/identity_embeds', data=embeddings.astype(np.float32))
		else:
			if 'poseest/identity_embeds' not in out_file:
				raise InvalidPoseFileException('Identity embedding values not provided and is required.')

	adjust_pose_version(pose_file, 4)


def write_v6_tracklets(pose_file, segmentation_tracks: np.ndarray, segmentation_ids: np.ndarray):
	"""Writes the optional segmentation tracklet and identity fields.

	Args:
		pose_file: file to write the data to
		segmentation_tracks: segmentation track data of shape [frame, num_animals]
		segmentation_ids: segmentation longterm id data of shape [frame, num_animals]

	Raises:
		InvalidPoseFileException if segmentation data is not present in the file or data is the wrong shape.
	"""
	with h5py.File(pose_file, 'a') as out_file:
		if 'poseest/seg_data' not in out_file:
			raise InvalidPoseFileException('Segmentation data not present in the file.')
		seg_shape = out_file['poseest/seg_data'].shape[:2]
		if segmentation_tracks.shape != seg_shape:
			raise InvalidPoseFileException('Segmentation track data does not match segmentation data shape.')
		if segmentation_ids.shape != seg_shape:
			raise InvalidPoseFileException('Segmentation identity data does not match segmentation data shape.')

		if 'poseest/instance_seg_id' in out_file:
			del out_file['poseest/instance_seg_id']
		out_file.create_dataset('poseest/instance_seg_id', data=segmentation_tracks.astype(np.uint32))
		if 'poseest/longterm_seg_id' in out_file:
			del out_file['poseest/longterm_seg_id']
		out_file.create_dataset('poseest/longterm_seg_id', data=segmentation_ids.astype(np.uint32))


def write_identity_data(pose_file, embeddings: np.ndarray, config_str: str = 'MNAS_latent16', model_str: str = '2022-04-28_model.ckpt-183819'):
	"""Writes identity prediction data to a pose file.

	Args:
		pose_file: file to write the data to
		embeddings: embedding data of shape [frame, n_animals, embed_dim]
		config_str: string defining the configuration of the model used
		model_str: string defining the checkpoint used

	Raises:
		InvalidPoseFileException if embedding shapes don't match pose in file.
	"""
	with h5py.File(pose_file, 'a') as out_file:
		if out_file['poseest/points'].shape[:2] != embeddings.shape[:2]:
			raise InvalidPoseFileException('Keypoint data does not match embedding data shape.')
		if 'poseest/identity_embeds' in out_file:
			del out_file['poseest/identity_embeds']
		out_file.create_dataset('poseest/identity_embeds', data=embeddings.astype(np.float32))
		out_file['poseest/identity_embeds'].attrs['config'] = config_str
		out_file['poseest/identity_embeds'].attrs['model'] = model_str

	adjust_pose_version(pose_file, 4)


def write_seg_data(pose_file, seg_contours_matrix: np.ndarray, seg_external_flags: np.ndarray, config_str: str = '', model_str: str = '', skip_matching: bool = False):
	"""Writes segmentation data to a pose file.

	Args:
		pose_file: file to write the data to
		seg_contours_matrix: contour data for segmentation of shape [frame, n_animals, n_contours, max_contour_length, 2]
		seg_external_flags: external flags for each contour of shape [frame, n_animals, n_contours]
		config_str: string defining the configuration of the model used
		model_str: string defining the checkpoint used
		skip_matching: boolean to skip matching (e.g. for topdown). Pose file will appear as though it does not contain segmentation data.

	Note:
		This function will automatically match segmentation data with pose data when `adjust_pose_version` is called.

	Raises:
		InvalidPoseFileException if shapes don't match
	"""
	if np.any(np.asarray(seg_contours_matrix.shape)[:3] != np.asarray(seg_external_flags.shape)):
		raise InvalidPoseFileException(f'Segmentation data shape does not match. Contour Shape: {seg_contours_matrix.shape}, Flag Shape: {seg_external_flags.shape}')

	with h5py.File(pose_file, 'a') as out_file:
		if 'poseest/seg_data' in out_file:
			del out_file['poseest/seg_data']
		out_file.create_dataset('poseest/seg_data', data=seg_contours_matrix, compression="gzip", compression_opts=9)
		out_file['poseest/seg_data'].attrs['config'] = config_str
		out_file['poseest/seg_data'].attrs['model'] = model_str
		if 'poseest/seg_external_flag' in out_file:
			del out_file['poseest/seg_external_flag']
		out_file.create_dataset('poseest/seg_external_flag', data=seg_external_flags, compression="gzip", compression_opts=9)

	if not skip_matching:
		adjust_pose_version(pose_file, 6)


def write_static_object_data(pose_file, object_data: np.ndarray, static_object: str, config_str: str = '', model_str: str = ''):
	"""Writes segmentation data to a pose file.

	Args:
		pose_file: file to write the data to
		object_data: static object data
		static_object: name of object
		config_str: string defining the configuration of the model used
		model_str: string defining the checkpoint used
	"""
	with h5py.File(pose_file, 'a') as out_file:
		if 'static_objects' in out_file and static_object in out_file['static_objects']:
			del out_file['static_objects/' + static_object]
		out_file.create_dataset('static_objects/' + static_object, data=object_data)
		out_file['static_objects/' + static_object].attrs['config'] = config_str
		out_file['static_objects/' + static_object].attrs['model'] = model_str

	adjust_pose_version(pose_file, 5)


def write_pixel_per_cm_attr(pose_file, px_per_cm: float, source: str):
	"""Writes pixel per cm data.

	Args:
		pose_file: file to write the data to
		px_per_cm: coefficient for converting pixels to cm
		source: string describing the source of this conversion
	"""
	with h5py.File(pose_file, 'a') as out_file:
		out_file['poseest'].attrs['cm_per_pixel'] = px_per_cm
		out_file['poseest'].attrs['cm_per_pixel_source'] = source
