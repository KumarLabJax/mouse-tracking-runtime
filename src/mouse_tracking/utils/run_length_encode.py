"""Run-Length Encoding Utility."""

import warnings

import numpy as np


def run_length_encode(
    input_array: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Perform run-length encoding on a 1-dimensional array.

    Run-length encoding compresses sequences of identical consecutive values
    into triplets of (start_position, duration, value).

    Args:
        input_array: A 1-dimensional numpy array to encode.

    Returns:
        A tuple containing three arrays:
        - start_positions: Starting indices of each run (None if input is empty)
        - durations: Length of each run (None if input is empty)
        - values: The value for each run (None if input is empty)

    Raises:
        ValueError: If input_array is not 1-dimensional.

    Examples:
        >>> arr = np.array([1, 1, 2, 2, 2, 3])
        >>> starts, durations, values = run_length_encode(arr)
        >>> print(starts)    # [0 2 5]
        >>> print(durations) # [2 3 1]
        >>> print(values)    # [1 2 3]

        >>> empty_arr = np.array([])
        >>> run_length_encode(empty_arr)
        (None, None, None)
    """
    # Convert input to numpy array and validate
    array = np.asarray(input_array)

    if array.ndim != 1:
        raise ValueError(f"Input must be 1-dimensional, got {array.ndim}D array")

    array_length = len(array)

    # Handle empty array case
    if array_length == 0:
        return None, None, None

    # Handle single element case
    if array_length == 1:
        return (np.array([0]), np.array([1]), np.array([array[0]]))

    # Find positions where consecutive elements differ
    change_mask = array[1:] != array[:-1]

    # Get indices of run endings (last index of each run)
    run_end_indices = np.append(np.where(change_mask)[0], array_length - 1)

    # Calculate run durations
    run_durations = np.diff(np.append(-1, run_end_indices))

    # Calculate run start positions
    run_start_positions = np.cumsum(np.append(0, run_durations))[:-1]

    # Get the values for each run
    run_values = array[run_end_indices]

    return run_start_positions, run_durations, run_values


def rle(
    inarray: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Backward compatibility alias for run_length_encode.

    Args:
        inarray: A 1-dimensional numpy array to encode.

    Returns:
        A tuple of (start_positions, durations, values).
    """
    # TODO: deprecate this function in favor of find_first_nonzero_index
    warnings.warn(
        "`rle` is deprecated, use `run_length_encode` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # return run_length_encode(inarray)
    ia = np.asarray(inarray)
    n = len(ia)
    if n == 0:
        return (None, None, None)
    y = ia[1:] != ia[:-1]
    i = np.append(np.where(y), n - 1)
    z = np.diff(np.append(-1, i))
    p = np.cumsum(np.append(0, z))[:-1]
    return (p, z, ia[i])
