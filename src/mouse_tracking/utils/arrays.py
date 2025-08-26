"""Numpy array utility functions for mouse tracking."""

import warnings

import cv2
import numpy as np


def find_first_nonzero_index(array: np.ndarray) -> int:
    """
    Find the index of the first non-zero element in an array.

    This function searches through the array and returns the index of the first
    element that evaluates to True (non-zero for numeric types, True for booleans,
    non-empty for strings, etc.).

    Args:
        array: A numpy array to search through. Can be of any numeric type,
               boolean, or other type that supports truthiness evaluation.

    Returns:
        The index (int) of the first non-zero/truthy element in the array.
        Returns -1 if no non-zero elements are found or if the array is empty.

    Raises:
        TypeError: If the input cannot be converted to a numpy array.

    Examples:
        >>> arr = np.array([0, 0, 5, 3, 0])
        >>> find_first_nonzero_index(arr)
        2

        >>> arr = np.array([0, 0, 0])
        >>> find_first_nonzero_index(arr)
        -1

        >>> arr = np.array([1, 2, 3])
        >>> find_first_nonzero_index(arr)
        0

        >>> arr = np.array([])
        >>> find_first_nonzero_index(arr)
        -1

        >>> arr = np.array([False, True, False])
        >>> find_first_nonzero_index(arr)
        1
    """
    try:
        # Convert input to numpy array
        input_array = np.asarray(array)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Input cannot be converted to numpy array: {e}") from e

    # Handle empty array case
    if input_array.size == 0:
        return -1

    # Find indices of non-zero elements
    nonzero_indices = np.where(input_array)[0]

    # Return first index if any non-zero elements exist, otherwise -1
    if nonzero_indices.size == 0:
        return -1

    # np.where returns indices in sorted order for 1D arrays, so first element is minimum
    return int(nonzero_indices[0])


def safe_find_first(arr: np.ndarray):
    """Finds the first non-zero index in an array.

    Args:
            arr: array to search

    Returns:
            integer index of the first non-zero element, -1 if no non-zero elements
    """
    # TODO: deprecate this function in favor of find_first_nonzero_index
    warnings.warn(
        "`safe_find_first` is deprecated, use `find_first_nonzero_index` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # return find_first_nonzero_index(arr)

    nonzero = np.where(arr)[0]
    if len(nonzero) == 0:
        return -1
    return sorted(nonzero)[0]


def argmax_2d(arr: np.ndarray):
    """Obtains the peaks for all keypoints in a pose.

    Args:
            arr: np.ndarray of shape [batch, 12, img_width, img_height]

    Returns:
            tuple of (values, coordinates)
            values: array of shape [batch, 12] containing the maximal values per-keypoint
            coordinates: array of shape [batch, 12, 2] containing the coordinates
    """
    full_max_cols = np.argmax(arr, axis=-1, keepdims=True)
    max_col_vals = np.take_along_axis(arr, full_max_cols, axis=-1)
    max_rows = np.argmax(max_col_vals, axis=-2, keepdims=True)
    max_row_vals = np.take_along_axis(max_col_vals, max_rows, axis=-2)
    max_cols = np.take_along_axis(full_max_cols, max_rows, axis=-2)

    max_vals = max_row_vals.squeeze(-1).squeeze(-1)
    max_idxs = np.stack(
        [max_rows.squeeze(-1).squeeze(-1), max_cols.squeeze(-1).squeeze(-1)], axis=-1
    )

    return max_vals, max_idxs


def get_peak_coords(arr: np.ndarray):
    """Converts a boolean array of peaks into locations.

    Args:
            arr: array of shape [w, h] to search for peaks

    Returns:
            tuple of (values, coordinates)
            values: array of shape [n_peaks] containing the maximal values per-peak
            coordinates: array of shape [n_peaks, 2] containing the coordinates
    """
    peak_locations = np.argwhere(arr)
    if len(peak_locations) == 0:
        return np.zeros([0], dtype=np.float32), np.zeros([0, 2], dtype=np.int16)

    max_vals = [arr[coord.tolist()] for coord in peak_locations]

    return np.stack(max_vals), peak_locations


def localmax_2d(arr: np.ndarray, threshold: int | float, radius: int | float):
    """Obtains the multiple peaks with non-max suppression.

    Args:
            arr: np.ndarray of shape [img_width, img_height]
            threshold: threshold required for a positive to be found
            radius: square radius (rectangle, not circle) peaks must be apart to be
                     considered a peak. Largest peaks will cause all other potential peaks
                     in this radius to be omitted.

    Returns:
            tuple of (values, coordinates)
            values: array of shape [n_peaks] containing the maximal values per-peak
            coordinates: array of shape [n_peaks, 2] containing the coordinates
    """
    assert radius >= 1
    assert np.squeeze(arr).ndim == 2

    point_heatmap = np.expand_dims(np.squeeze(arr), axis=-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius * 2 + 1, radius * 2 + 1))
    # Non-max suppression
    dilated = cv2.dilate(point_heatmap, kernel)
    mask = arr >= dilated
    eroded = cv2.erode(point_heatmap, kernel)
    mask_2 = arr > eroded
    mask = np.logical_and(mask, mask_2)
    # Peakfinding via Threshold
    mask = np.logical_and(mask, arr > threshold)
    bool_arr = np.full(dilated.shape, False, dtype=bool)
    bool_arr[mask] = True
    return get_peak_coords(bool_arr)
