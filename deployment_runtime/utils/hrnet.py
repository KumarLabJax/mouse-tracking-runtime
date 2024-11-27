import torch
import torchvision.transforms as transforms


def argmax_2d_torch(tensor):
	"""Obtains the beaks for all keypoints in a pose.

	Args:
		tensor: pytorch tensor of shape [batch, 12, img_width, img_height]

	Returns:
		tuple of (values, coordinates)
		values: array of shape [batch, 12] containing the maximal values per-keypoint
		coordinates: array of shape [batch, 12, 2] containing the coordinates
	"""
	assert tensor.dim() >= 2
	max_col_vals, max_cols = torch.max(tensor, -1, keepdim=True)
	max_vals, max_rows = torch.max(max_col_vals, -2, keepdim=True)
	max_cols = torch.gather(max_cols, -2, max_rows)
	
	max_vals = max_vals.squeeze(-1).squeeze(-1)
	max_rows = max_rows.squeeze(-1).squeeze(-1)
	max_cols = max_cols.squeeze(-1).squeeze(-1)
	
	return max_vals, torch.stack([max_rows, max_cols], -1)


def preprocess_hrnet():
	"""Preprocess transformation for hrnet.

	Retuns:
		transform function which can be called on batch data
	"""
	xform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.45, 0.45, 0.45],
			std=[0.225, 0.225, 0.225],
		),
	])
	return xform
