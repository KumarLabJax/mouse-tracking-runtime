import cv2
import numpy as np


def get_contours(
    mask_img: np.ndarray, min_contour_area: float = 50.0
) -> list[np.ndarray]:
    """Creates an opencv-complaint contour list given a mask.

    Args:
            mask_img: binary image of shape [width, height]
            min_contour_area: contours below this area are discarded

    Returns:
            Tuple of (contours, heirarchy)
            contours: Opencv-complains list of contours
            heirarchy: Opencv contour heirarchy
    """
    if np.any(mask_img):
        contours, tree = cv2.findContours(
            mask_img.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if min_contour_area > 0:
            contours_to_keep = []
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) > min_contour_area:
                    contours_to_keep.append(i)
            if len(contours_to_keep) > 0:
                contours = [contours[x] for x in contours_to_keep]
                tree = tree[0, np.array(contours_to_keep), :].reshape([1, -1, 4])
            else:
                contours = []
        if len(contours) > 0:
            return contours, tree
    return [np.zeros([0, 2], dtype=np.int32)], [np.zeros([0, 4], dtype=np.int32)]


def pad_contours(contours: list[np.ndarray], default_val: int = -1) -> np.ndarray:
    """Converts a list of contour data into a padded full matrix.

    Args:
            contours: Opencv-complaint contour data
            default_val: value used for padding

    Returns:
            Contour data in a padded matrix of shape [n_contours, n_points, 2]
    """
    num_contours = len(contours)
    max_contour_length = np.max([len(x) for x in contours])

    padded_matrix = np.full(
        [num_contours, max_contour_length, 2], default_val, dtype=np.int32
    )
    for i, cur_contour in enumerate(contours):
        padded_matrix[i, : cur_contour.shape[0], :] = np.squeeze(cur_contour)

    return padded_matrix


def merge_multiple_seg_instances(
    matrix_list: list[np.ndarray], flag_list: list[np.ndarray], default_val: int = -1
):
    """Merges multiple segmentation predictions together.

    Args:
            matrix_list: list of padded contour matrix
            flag_list: list of external flags
            default_val: value to pad full matrix with

    Returns:
            tuple of (segmentation_data, flag_data)
            segmentation_data: padded contour matrix containing all instances
            flag_data: padded flag matrix containing all flags

    Raises:
            AssertionError if the same number of predictions are not provided.
    """
    assert len(matrix_list) == len(flag_list)

    matrix_shapes = np.asarray([x.shape for x in matrix_list])

    # No predictions, just return default data containing smallest pads
    if len(matrix_shapes) == 0:
        return np.full([1, 1, 1, 2], default_val, dtype=np.int32), np.full(
            [1, 1], default_val, dtype=np.int32
        )

    flag_shapes = np.asarray([x.shape for x in flag_list])
    n_predictions = len(matrix_list)

    padded_matrix = np.full(
        [n_predictions, *np.max(matrix_shapes, axis=0).tolist()],
        default_val,
        dtype=np.int32,
    )
    padded_flags = np.full(
        [n_predictions, *np.max(flag_shapes, axis=0).tolist()],
        default_val,
        dtype=np.int32,
    )

    for i in range(n_predictions):
        dim1, dim2, dim3 = matrix_list[i].shape
        # No segmentation data, just skip it
        if dim2 == 0:
            continue
        padded_matrix[i, :dim1, :dim2, :dim3] = matrix_list[i]
        padded_flags[i, :dim1] = flag_list[i]

    return padded_matrix, padded_flags


def get_trimmed_contour(padded_contour, default_val=-1):
    """Removes padding from contour data.

    Args:
            padded_contour: a matrix of shape [n_points, 2] that has been padded
            default_val: pad value in the matrix

    Returns:
            an opencv-compliant contour
    """
    mask = np.all(padded_contour == default_val, axis=1)
    trimmed_contour = np.reshape(padded_contour[~mask, :], [-1, 2])
    return trimmed_contour.astype(np.int32)


def get_contour_stack(contour_mat, default_val=-1):
    """Helper function to return a contour list.

    Args:
            contour_mat: a full matrix of shape [n_contours, n_points, 2] or [n_points, 2] that contains a padded list of opencv contours
            default_val: pad value in the matrix

    Returns:
            an opencv-complaint contour list

    Raises:
            ValueError if shape of matrix is invalid

    Notes:
            Will always return a list of contours. This list may be of length 0
    """
    # Only one contour was stored per-mouse
    if np.ndim(contour_mat) == 2:
        trimmed_contour = get_trimmed_contour(contour_mat, default_val)
        contour_stack = [trimmed_contour]
    # Entire contour list was stored
    elif np.ndim(contour_mat) == 3:
        contour_stack = []
        for part_idx in np.arange(np.shape(contour_mat)[0]):
            cur_contour = contour_mat[part_idx]
            if np.all(cur_contour == default_val):
                break
            trimmed_contour = get_trimmed_contour(cur_contour, default_val)
            contour_stack.append(trimmed_contour)
    elif contour_mat is None:
        contour_stack = []
    else:
        raise ValueError("Contour matrix invalid")
    return contour_stack


def get_frame_masks(contour_mat, frame_size=None):
    """Returns a stack of masks for all valid contours.

    Args:
            contour_mat: a contour matrix of shape [n_animals, n_contours, n_points, 2]
            frame_size: frame size to render the contours on

    Returns:
            a stack of rendered contour masks
    """
    if frame_size is None:
        frame_size = [800, 800]
    frame_stack = []
    for animal_idx in np.arange(np.shape(contour_mat)[0]):
        new_frame = render_blob(contour_mat[animal_idx], frame_size=frame_size)
        frame_stack.append(new_frame.astype(bool))
    if len(frame_stack) > 0:
        return np.stack(frame_stack)
    return np.zeros([0, frame_size[0], frame_size[1]])


def render_blob(contour, frame_size=None, default_val=-1):
    """Renders a mask for an individual.

    Args:
            contour: a padded contour matrix of shape [n_contours, n_points, 2] or [n_points, 2]
            frame_size: frame size to render the contour
            default_val: pad value in the contour matrix

    Returns:
            boolean image of the rendered mask
    """
    if frame_size is None:
        frame_size = [800, 800]
    new_mask = np.zeros(frame_size, dtype=np.uint8)
    contour_stack = get_contour_stack(contour, default_val=default_val)
    # Note: We need to plot them all at the same time to have opencv properly detect holes
    _ = cv2.drawContours(new_mask, contour_stack, -1, (1), thickness=cv2.FILLED)
    return new_mask.astype(bool)


def get_frame_outlines(contour_mat, frame_size=None, thickness=1):
    """Renders a stack of outlines for all valid contours.

    Args:
            contour_mat: a contour matrix of shape [n_animals, n_contours, n_points, 2]
            frame_size: frame size to render the contours on
            thickness: thickness of the contour outline

    Returns:
            a stack of rendered outlines
    """
    if frame_size is None:
        frame_size = [800, 800]
    frame_stack = []
    for animal_idx in np.arange(np.shape(contour_mat)[0]):
        new_frame = render_outline(
            contour_mat[animal_idx], frame_size=frame_size, thickness=thickness
        )
        frame_stack.append(new_frame.astype(bool))
    if len(frame_stack) > 0:
        return np.stack(frame_stack)
    return np.zeros([0, frame_size[0], frame_size[1]])


def render_outline(contour, frame_size=None, thickness=1, default_val=-1):
    """Renders a mask outline for an individual.

    Args:
            contour: a padded contour matrix of shape [n_contours, n_points, 2] or [n_points, 2]
            frame_size: frame size to render the contour
            thickness: thickness of the contour outline
            default_val: pad value in the contour matrix

    Returns:
            boolean image of the rendered mask outline
    """
    if frame_size is None:
        frame_size = [800, 800]
    new_mask = np.zeros(frame_size, dtype=np.uint8)
    contour_stack = get_contour_stack(contour)
    # Note: We need to plot them all at the same time to have opencv properly detect holes
    _ = cv2.drawContours(new_mask, contour_stack, -1, (1), thickness=thickness)
    return new_mask.astype(bool)


def render_segmentation_overlay(
    contour, image, color: tuple[int] = (0, 0, 255)
) -> np.ndarray:
    """Renders segmentation contour data onto a frame.

    Args:
            contour: a padded contour matrix of shape [n_contours, n_points, 2] or [n_points, 2]
            image: image to render the contour onto
            color: color to render the outline of the contour

    Returns:
            copy of the image with the contour rendered
    """
    if np.all(contour == -1):
        return image
    outline = render_outline(contour, frame_size=image.shape[:2])
    new_image = image.copy()
    if new_image.shape[2] == 1:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2RGB)
    new_image[outline] = color
    return new_image
