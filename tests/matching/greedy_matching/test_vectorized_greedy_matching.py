"""Tests for vectorized_greedy_matching function."""

import numpy as np

from mouse_tracking.matching.greedy_matching import vectorized_greedy_matching


class TestVectorizedGreedyMatching:
    """Test basic functionality of vectorized_greedy_matching."""

    def test_basic_matching(self):
        """Test basic greedy matching functionality."""
        # Create a simple cost matrix
        cost_matrix = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0], [7.0, 8.0, 1.5]])
        max_cost = 10.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Should be a dictionary mapping column indices to row indices
        assert isinstance(matches, dict)

        # Check that matches are valid
        for col_idx, row_idx in matches.items():
            assert 0 <= col_idx < cost_matrix.shape[1]
            assert 0 <= row_idx < cost_matrix.shape[0]
            assert cost_matrix[row_idx, col_idx] < max_cost

        # Check that no row or column is used twice
        used_rows = set(matches.values())
        used_cols = set(matches.keys())
        assert len(used_rows) == len(matches)  # No duplicate rows
        assert len(used_cols) == len(matches)  # No duplicate columns

    def test_greedy_selects_lowest_cost(self):
        """Test that greedy algorithm selects lowest cost matches first."""
        # Create a cost matrix where the optimal greedy choice is clear
        cost_matrix = np.array([[1.0, 10.0], [10.0, 2.0]])
        max_cost = 15.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Should match (0,0) and (1,1) since these have lowest costs
        assert matches == {0: 0, 1: 1}

    def test_max_cost_threshold(self):
        """Test that max_cost threshold is respected."""
        cost_matrix = np.array([[1.0, 5.0, 15.0], [8.0, 2.0, 20.0], [12.0, 18.0, 3.0]])
        max_cost = 10.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # All matches should have cost < max_cost
        for col_idx, row_idx in matches.items():
            assert cost_matrix[row_idx, col_idx] < max_cost

        # Should not match any costs >= max_cost
        for col_idx, row_idx in matches.items():
            assert cost_matrix[row_idx, col_idx] != 15.0
            assert cost_matrix[row_idx, col_idx] != 20.0
            assert cost_matrix[row_idx, col_idx] != 12.0
            assert cost_matrix[row_idx, col_idx] != 18.0

    def test_empty_matrix_handling(self):
        """Test handling of empty matrices."""
        # Empty matrix (0x0)
        cost_matrix = np.array([]).reshape(0, 0)
        matches = vectorized_greedy_matching(cost_matrix, 10.0)
        assert matches == {}

        # Empty rows (0x3)
        cost_matrix = np.array([]).reshape(0, 3)
        matches = vectorized_greedy_matching(cost_matrix, 10.0)
        assert matches == {}

        # Empty columns (3x0)
        cost_matrix = np.array([]).reshape(3, 0)
        matches = vectorized_greedy_matching(cost_matrix, 10.0)
        assert matches == {}

    def test_single_element_matrix(self):
        """Test with single element matrix."""
        cost_matrix = np.array([[5.0]])

        # Should match if cost < max_cost
        matches = vectorized_greedy_matching(cost_matrix, 10.0)
        assert matches == {0: 0}

        # Should not match if cost >= max_cost
        matches = vectorized_greedy_matching(cost_matrix, 3.0)
        assert matches == {}

    def test_no_valid_matches(self):
        """Test when no matches are below max_cost threshold."""
        cost_matrix = np.array([[15.0, 20.0], [25.0, 30.0]])
        max_cost = 10.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)
        assert matches == {}

    def test_rectangular_matrices(self):
        """Test with non-square matrices."""
        # More rows than columns
        cost_matrix = np.array([[1.0, 5.0], [2.0, 3.0], [4.0, 6.0]])
        max_cost = 10.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Should have at most min(n_rows, n_cols) matches
        assert len(matches) <= min(cost_matrix.shape)

        # Check validity
        for col_idx, row_idx in matches.items():
            assert cost_matrix[row_idx, col_idx] < max_cost

        # More columns than rows
        cost_matrix = np.array([[1.0, 5.0, 3.0, 7.0], [2.0, 4.0, 6.0, 8.0]])
        max_cost = 10.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Should have at most min(n_rows, n_cols) matches
        assert len(matches) <= min(cost_matrix.shape)

        # Check validity
        for col_idx, row_idx in matches.items():
            assert cost_matrix[row_idx, col_idx] < max_cost


class TestVectorizedGreedyMatchingEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_identical_costs(self):
        """Test behavior with identical costs."""
        cost_matrix = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
        max_cost = 10.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Should still produce valid matches
        assert len(matches) == min(cost_matrix.shape)
        for col_idx, row_idx in matches.items():
            assert cost_matrix[row_idx, col_idx] == 5.0

    def test_inf_and_nan_costs(self):
        """Test handling of infinite and NaN costs."""
        cost_matrix = np.array(
            [[1.0, np.inf, 3.0], [np.nan, 2.0, np.inf], [4.0, 5.0, np.nan]]
        )
        max_cost = 10.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Should only match finite costs < max_cost
        for col_idx, row_idx in matches.items():
            cost = cost_matrix[row_idx, col_idx]
            assert np.isfinite(cost)
            assert cost < max_cost

    def test_negative_costs(self):
        """Test handling of negative costs."""
        cost_matrix = np.array([[-1.0, 5.0, 3.0], [2.0, -2.0, 6.0], [4.0, 8.0, -0.5]])
        max_cost = 10.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Should prefer negative costs (lowest first)
        # Expected matches: (-2.0, -1.0, -0.5) would be preferred
        matched_costs = [
            cost_matrix[row_idx, col_idx] for col_idx, row_idx in matches.items()
        ]

        # Should include negative costs
        assert any(cost < 0 for cost in matched_costs)

        # All should be valid
        for col_idx, row_idx in matches.items():
            assert cost_matrix[row_idx, col_idx] < max_cost

    def test_zero_max_cost(self):
        """Test with zero max_cost."""
        cost_matrix = np.array([[1.0, -1.0], [-2.0, 0.5]])
        max_cost = 0.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Should only match costs < 0
        for col_idx, row_idx in matches.items():
            assert cost_matrix[row_idx, col_idx] < 0.0

    def test_negative_max_cost(self):
        """Test with negative max_cost."""
        cost_matrix = np.array([[-1.0, 5.0], [-3.0, 2.0]])
        max_cost = -2.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Should only match costs < -2.0
        for col_idx, row_idx in matches.items():
            assert cost_matrix[row_idx, col_idx] < -2.0

    def test_large_matrices(self):
        """Test performance with larger matrices."""
        # Create a larger matrix
        n = 100
        np.random.seed(42)  # For reproducibility
        cost_matrix = np.random.random((n, n)) * 10
        max_cost = 5.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Should still produce valid matches
        for col_idx, row_idx in matches.items():
            assert 0 <= col_idx < n
            assert 0 <= row_idx < n
            assert cost_matrix[row_idx, col_idx] < max_cost

        # Should not have duplicate assignments
        assert len(set(matches.values())) == len(matches)
        assert len(set(matches.keys())) == len(matches)


class TestVectorizedGreedyMatchingAlgorithmProperties:
    """Test algorithmic properties and correctness."""

    def test_greedy_property(self):
        """Test that algorithm follows greedy property (lowest cost first)."""
        cost_matrix = np.array([[5.0, 1.0, 3.0], [2.0, 4.0, 6.0], [8.0, 7.0, 9.0]])
        max_cost = 10.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Get matched costs
        matched_costs = []
        for col_idx, row_idx in matches.items():
            matched_costs.append(cost_matrix[row_idx, col_idx])

        # Should include the lowest cost (1.0)
        assert 1.0 in matched_costs

        # Should not include higher costs if lower ones are available
        # Given the greedy nature, cost 1.0 should be matched first
        if 1.0 in matched_costs:
            # Column 1 should be matched to row 0
            assert matches.get(1) == 0

    def test_optimal_vs_greedy(self):
        """Test case where greedy solution differs from optimal."""
        # Create a case where greedy != optimal
        cost_matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
        max_cost = 10.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Greedy should pick the globally minimum cost first (1.0)
        # Both (0,0) and (1,1) have cost 1.0, but algorithm picks first occurrence
        matched_costs = [
            cost_matrix[row_idx, col_idx] for col_idx, row_idx in matches.items()
        ]

        # Should have 2 matches, both with cost 1.0 or 2.0
        assert len(matches) == 2
        assert all(cost <= 2.0 for cost in matched_costs)

    def test_matching_uniqueness(self):
        """Test that each row and column is used at most once."""
        cost_matrix = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        max_cost = 10.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Each row and column should be used exactly once
        assert len(set(matches.values())) == len(matches)  # Unique rows
        assert len(set(matches.keys())) == len(matches)  # Unique columns
        assert len(matches) == min(cost_matrix.shape)

    def test_cost_ordering(self):
        """Test that matches are processed in cost order."""
        cost_matrix = np.array([[3.0, 1.0, 5.0], [6.0, 2.0, 4.0], [9.0, 8.0, 7.0]])
        max_cost = 10.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # The algorithm should process in order: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
        # So (0,1) should be matched first (cost 1.0)
        # Then (1,1) cannot be matched (column 1 used), so (1,0) might be next available

        # At minimum, the lowest cost should be matched
        matched_costs = [
            cost_matrix[row_idx, col_idx] for col_idx, row_idx in matches.items()
        ]
        assert 1.0 in matched_costs  # Lowest cost should be matched

    def test_collision_handling(self):
        """Test that row/column collisions are handled correctly."""
        # Create a matrix where multiple low costs compete for same row/column
        cost_matrix = np.array([[1.0, 2.0, 10.0], [3.0, 1.0, 10.0], [10.0, 10.0, 1.0]])
        max_cost = 5.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Should handle conflicts correctly
        # Costs 1.0 appear at (0,0), (1,1), (2,2)
        # All should be matchable since they don't conflict
        assert len(matches) == 3

        # Check that all matches are the 1.0 costs
        for col_idx, row_idx in matches.items():
            assert cost_matrix[row_idx, col_idx] == 1.0


class TestVectorizedGreedyMatchingDataTypes:
    """Test data type handling and validation."""

    def test_integer_costs(self):
        """Test with integer cost matrices."""
        cost_matrix = np.array([[1, 5, 3], [4, 2, 6], [7, 8, 1]], dtype=int)
        max_cost = 10

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Should work with integers
        assert isinstance(matches, dict)
        for col_idx, row_idx in matches.items():
            assert cost_matrix[row_idx, col_idx] < max_cost

    def test_float32_costs(self):
        """Test with float32 cost matrices."""
        cost_matrix = np.array(
            [[1.0, 5.0, 3.0], [4.0, 2.0, 6.0], [7.0, 8.0, 1.0]], dtype=np.float32
        )
        max_cost = 10.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Should work with float32
        assert isinstance(matches, dict)
        for col_idx, row_idx in matches.items():
            assert cost_matrix[row_idx, col_idx] < max_cost

    def test_different_max_cost_types(self):
        """Test with different max_cost data types."""
        cost_matrix = np.array([[1.0, 5.0], [4.0, 2.0]])

        # Test with int max_cost
        matches = vectorized_greedy_matching(cost_matrix, 10)
        assert len(matches) > 0

        # Test with float max_cost
        matches = vectorized_greedy_matching(cost_matrix, 10.0)
        assert len(matches) > 0

        # Test with numpy scalar max_cost
        matches = vectorized_greedy_matching(cost_matrix, np.float64(10.0))
        assert len(matches) > 0


class TestVectorizedGreedyMatchingPerformance:
    """Test performance characteristics and complexity."""

    def test_sparse_matrix_performance(self):
        """Test performance with sparse valid costs."""
        # Create a matrix where most costs are too high
        n = 50
        cost_matrix = np.full((n, n), 1000.0)  # High costs everywhere

        # Add a few valid low costs
        np.random.seed(42)
        for _ in range(10):
            i, j = np.random.randint(0, n, 2)
            cost_matrix[i, j] = np.random.random() * 5.0

        max_cost = 10.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Should only match the low costs
        assert len(matches) <= 10
        for col_idx, row_idx in matches.items():
            assert cost_matrix[row_idx, col_idx] < max_cost

    def test_dense_matrix_performance(self):
        """Test performance with dense valid costs."""
        # Create a matrix where most costs are valid
        n = 50
        np.random.seed(42)
        cost_matrix = np.random.random((n, n)) * 5.0  # All costs < 10.0
        max_cost = 10.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Should match up to min(n, n) = n pairs
        assert len(matches) == n
        for col_idx, row_idx in matches.items():
            assert cost_matrix[row_idx, col_idx] < max_cost

    def test_benchmark_timing(self):
        """Basic timing test to ensure reasonable performance."""
        # Create a moderately sized matrix
        n = 100
        np.random.seed(42)
        cost_matrix = np.random.random((n, n)) * 10.0
        max_cost = 5.0

        import time

        start_time = time.time()
        matches = vectorized_greedy_matching(cost_matrix, max_cost)
        end_time = time.time()

        # Should complete in reasonable time (< 1 second for 100x100)
        elapsed = end_time - start_time
        assert elapsed < 1.0, f"Function took {elapsed:.3f}s, expected < 1.0s"

        # Should produce valid results
        assert isinstance(matches, dict)
        for col_idx, row_idx in matches.items():
            assert cost_matrix[row_idx, col_idx] < max_cost


class TestVectorizedGreedyMatchingComparison:
    """Test comparison with expected results for known cases."""

    def test_textbook_example(self):
        """Test with a well-known assignment problem example."""
        # Classical assignment problem
        cost_matrix = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
        max_cost = 10.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Greedy should pick minimum cost (0) first, then next available minimums
        # Cost 0 is at (1,1), so column 1 and row 1 are used
        # Next minimum available would be 1 at (0,1) - but column 1 used
        # So next is 2 at (1,0) - but row 1 used
        # So next is 2 at (2,1) - but column 1 used
        # So next is 2 at (2,2)
        # etc.

        matched_costs = [
            cost_matrix[row_idx, col_idx] for col_idx, row_idx in matches.items()
        ]

        # Should include the minimum cost
        assert 0 in matched_costs

        # Should have 3 matches (square matrix)
        assert len(matches) == 3

    def test_known_optimal_case(self):
        """Test case where greedy solution is optimal."""
        cost_matrix = np.array([[1, 9, 9], [9, 2, 9], [9, 9, 3]])
        max_cost = 10.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Greedy should find optimal solution: (0,0), (1,1), (2,2)
        expected_matches = {0: 0, 1: 1, 2: 2}
        assert matches == expected_matches

    def test_suboptimal_greedy_case(self):
        """Test case where greedy finds optimal solution when costs don't conflict."""
        cost_matrix = np.array([[1, 2], [2, 1]])
        max_cost = 10.0

        matches = vectorized_greedy_matching(cost_matrix, max_cost)

        # Both 1's are processed first and don't conflict with each other
        # So greedy actually finds optimal solution: (0,0) and (1,1)
        assert len(matches) == 2

        matched_costs = [
            cost_matrix[row_idx, col_idx] for col_idx, row_idx in matches.items()
        ]
        total_cost = sum(matched_costs)

        # Should find optimal solution in this case
        assert total_cost == 2.0  # 1 + 1

        # Verify the actual matches
        expected_matches = {0: 0, 1: 1}
        assert matches == expected_matches
