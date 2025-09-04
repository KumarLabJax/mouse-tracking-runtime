import torch


def argmax_2d_torch(tensor):
    """Obtains the peaks for all keypoints in a pose.

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


def localmax_2d_torch(tensor, min_thresh, min_dist):
    """Obtains local peaks in a tensor.

    Args:
            tensor: pytorch tensor of shape [1, img_width, img_height] or [batch, 1, img_width, img_height]
            min_thresh: minimum value to be considered a peak
            min_dist: minimum distance away from another peak to still be considered a peak

    Returns:
            A boolean tensor where Trues indicate where a local maxima was detected.
    """
    assert min_dist >= 1
    # Make sure the data is the correct shape
    # Allow 3 (single image) or 4 (batched images)
    orig_dim = tensor.dim()
    if tensor.dim() == 3:
        tensor = torch.unsqueeze(tensor, 0)
    assert tensor.dim() == 4

    # Peakfinding
    dilated = torch.nn.MaxPool2d(
        kernel_size=min_dist * 2 + 1, stride=1, padding=min_dist
    )(tensor)
    mask = tensor >= dilated
    # Non-max suppression
    eroded = -torch.nn.MaxPool2d(
        kernel_size=min_dist * 2 + 1, stride=1, padding=min_dist
    )(-tensor)
    mask_2 = tensor > eroded
    mask = torch.logical_and(mask, mask_2)
    # Threshold
    mask = torch.logical_and(mask, tensor > min_thresh)
    bool_arr = torch.zeros_like(dilated, dtype=bool) + 1
    bool_arr[~mask] = 0
    if orig_dim == 3:
        bool_arr = torch.squeeze(bool_arr, 0)
    return bool_arr


def preprocess_hrnet(arr):
    """Preprocess transformation for hrnet.

    Args:
            arr: numpy array of shape [img_w, img_h, img_d]

    Retuns:
            pytorch tensor with hrnet transformations applied
    """
    # Original function was this:
    # xform = transforms.Compose([
    # 	transforms.ToTensor(),
    # 	transforms.Normalize(
    # 		mean=[0.45, 0.45, 0.45],
    # 		std=[0.225, 0.225, 0.225],
    # 	),
    # ])
    # ToTensor transform includes channel re-ordering and 0-255 to 0-1 scaling
    img_tensor = torch.tensor(arr)
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.unsqueeze(0).permute((0, 3, 1, 2))

    # Normalize transform
    mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor
