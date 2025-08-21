"""Optimized greedy matching algorithms for mouse tracking."""

import numpy as np


def vectorized_greedy_matching(cost_matrix: np.ndarray, max_cost: float) -> dict:
    """Optimized greedy matching using heap-based approach for O(k log k) complexity.

    This replaces the current O(n³) approach with a more efficient algorithm that:
    1. Pre-sorts all valid costs once: O(k log k) where k = number of valid costs
    2. Processes matches in cost order: O(k)
    3. Uses boolean arrays for O(1) collision detection

    Args:
            cost_matrix: Cost matrix of shape (n1, n2) with matching costs
            max_cost: Maximum cost threshold for valid matches

    Returns:
            Dictionary mapping column indices to row indices for matched pairs
    """
    n1, n2 = cost_matrix.shape
    matches = {}

    # Early return for empty matrices
    if n1 == 0 or n2 == 0:
        return matches

    # Find all valid costs and their indices
    valid_mask = cost_matrix < max_cost
    if not np.any(valid_mask):
        return matches

    # Extract valid costs and their coordinates
    valid_costs = cost_matrix[valid_mask]
    valid_indices = np.where(valid_mask)
    valid_rows = valid_indices[0]
    valid_cols = valid_indices[1]

    # Sort by cost (ascending)
    sorted_indices = np.argsort(valid_costs)

    # Track which rows and columns have been used
    used_rows = np.zeros(n1, dtype=bool)
    used_cols = np.zeros(n2, dtype=bool)

    # Process matches in cost order
    for idx in sorted_indices:
        row = valid_rows[idx]
        col = valid_cols[idx]

        # Check if both row and col are still available
        if not used_rows[row] and not used_cols[col]:
            matches[col] = row
            used_rows[row] = True
            used_cols[col] = True

    return matches
