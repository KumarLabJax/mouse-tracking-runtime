"""
Unit tests for PoseUtilsConfig from mouse_tracking.core.config.pose_utils.

This module tests the PoseUtilsConfig class which contains configuration values
for pose utility functions including keypoint indices, confidence thresholds,
mouse colors, and static object settings.
"""

from mouse_tracking.core.config.pose_utils import PoseUtilsConfig


class TestPoseUtilsConfigDefaults:
    """Test default values in PoseUtilsConfig."""

    def test_keypoint_indices_defaults(self):
        """Test that all keypoint indices have expected default values."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        assert config.NOSE_INDEX == 0
        assert config.LEFT_EAR_INDEX == 1
        assert config.RIGHT_EAR_INDEX == 2
        assert config.BASE_NECK_INDEX == 3
        assert config.LEFT_FRONT_PAW_INDEX == 4
        assert config.RIGHT_FRONT_PAW_INDEX == 5
        assert config.CENTER_SPINE_INDEX == 6
        assert config.LEFT_REAR_PAW_INDEX == 7
        assert config.RIGHT_REAR_PAW_INDEX == 8
        assert config.BASE_TAIL_INDEX == 9
        assert config.MID_TAIL_INDEX == 10
        assert config.TIP_TAIL_INDEX == 11

    def test_keypoint_indices_are_sequential(self):
        """Test that keypoint indices form a sequential range from 0 to 11."""
        # Arrange & Act
        config = PoseUtilsConfig()
        indices = [
            config.NOSE_INDEX,
            config.LEFT_EAR_INDEX,
            config.RIGHT_EAR_INDEX,
            config.BASE_NECK_INDEX,
            config.LEFT_FRONT_PAW_INDEX,
            config.RIGHT_FRONT_PAW_INDEX,
            config.CENTER_SPINE_INDEX,
            config.LEFT_REAR_PAW_INDEX,
            config.RIGHT_REAR_PAW_INDEX,
            config.BASE_TAIL_INDEX,
            config.MID_TAIL_INDEX,
            config.TIP_TAIL_INDEX,
        ]

        # Assert
        assert sorted(indices) == list(range(12)), (
            "Keypoint indices should be sequential from 0 to 11"
        )
        assert len(indices) == len(set(indices)), (
            "All keypoint indices should be unique"
        )

    def test_confidence_thresholds_defaults(self):
        """Test that confidence threshold values have expected defaults."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        assert config.MIN_HIGH_CONFIDENCE == 0.75
        assert config.MIN_GAIT_CONFIDENCE == 0.3
        assert config.MIN_JABS_CONFIDENCE == 0.3
        assert config.MIN_JABS_KEYPOINTS == 3

    def test_confidence_thresholds_are_valid(self):
        """Test that confidence thresholds are in valid ranges."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        assert 0.0 <= config.MIN_HIGH_CONFIDENCE <= 1.0, (
            "MIN_HIGH_CONFIDENCE should be between 0 and 1"
        )
        assert 0.0 <= config.MIN_GAIT_CONFIDENCE <= 1.0, (
            "MIN_GAIT_CONFIDENCE should be between 0 and 1"
        )
        assert 0.0 <= config.MIN_JABS_CONFIDENCE <= 1.0, (
            "MIN_JABS_CONFIDENCE should be between 0 and 1"
        )
        assert config.MIN_JABS_KEYPOINTS > 0, "MIN_JABS_KEYPOINTS should be positive"


class TestPoseUtilsConfigConnectedSegments:
    """Test CONNECTED_SEGMENTS configuration."""

    def test_connected_segments_structure(self):
        """Test that CONNECTED_SEGMENTS has expected structure."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        assert isinstance(config.CONNECTED_SEGMENTS, list), (
            "CONNECTED_SEGMENTS should be a list"
        )
        assert len(config.CONNECTED_SEGMENTS) == 3, (
            "CONNECTED_SEGMENTS should have 3 segments"
        )
        for segment in config.CONNECTED_SEGMENTS:
            assert isinstance(segment, list), "Each segment should be a list"
            assert all(isinstance(idx, int) for idx in segment), (
                "All segment indices should be integers"
            )

    def test_connected_segments_front_paws(self):
        """Test front paws segment connectivity."""
        # Arrange & Act
        config = PoseUtilsConfig()
        front_segment = config.CONNECTED_SEGMENTS[0]

        # Assert
        expected = [
            config.LEFT_FRONT_PAW_INDEX,
            config.CENTER_SPINE_INDEX,
            config.RIGHT_FRONT_PAW_INDEX,
        ]
        assert front_segment == expected, (
            "Front segment should connect left paw, spine, and right paw"
        )

    def test_connected_segments_rear_paws(self):
        """Test rear paws segment connectivity."""
        # Arrange & Act
        config = PoseUtilsConfig()
        rear_segment = config.CONNECTED_SEGMENTS[1]

        # Assert
        expected = [
            config.LEFT_REAR_PAW_INDEX,
            config.BASE_TAIL_INDEX,
            config.RIGHT_REAR_PAW_INDEX,
        ]
        assert rear_segment == expected, (
            "Rear segment should connect left paw, tail base, and right paw"
        )

    def test_connected_segments_spine_and_tail(self):
        """Test spine and tail segment connectivity."""
        # Arrange & Act
        config = PoseUtilsConfig()
        spine_tail_segment = config.CONNECTED_SEGMENTS[2]

        # Assert
        expected = [
            config.NOSE_INDEX,
            config.BASE_NECK_INDEX,
            config.CENTER_SPINE_INDEX,
            config.BASE_TAIL_INDEX,
            config.MID_TAIL_INDEX,
            config.TIP_TAIL_INDEX,
        ]
        assert spine_tail_segment == expected, (
            "Spine/tail segment should connect nose through tip of tail"
        )

    def test_connected_segments_indices_valid(self):
        """Test that all indices in CONNECTED_SEGMENTS are valid keypoint indices."""
        # Arrange & Act
        config = PoseUtilsConfig()
        all_indices = []
        for segment in config.CONNECTED_SEGMENTS:
            all_indices.extend(segment)

        # Assert
        for idx in all_indices:
            assert 0 <= idx <= 11, f"Index {idx} should be between 0 and 11"


class TestPoseUtilsConfigMouseColors:
    """Test MOUSE_COLORS configuration."""

    def test_mouse_colors_structure(self):
        """Test that MOUSE_COLORS has expected structure."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        assert isinstance(config.MOUSE_COLORS, list), "MOUSE_COLORS should be a list"
        assert len(config.MOUSE_COLORS) == 11, "MOUSE_COLORS should have 11 colors"

    def test_mouse_colors_are_valid_rgb_tuples(self):
        """Test that all mouse colors are valid RGB tuples."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        for i, color in enumerate(config.MOUSE_COLORS):
            assert isinstance(color, tuple), f"Color {i} should be a tuple"
            assert len(color) == 3, f"Color {i} should have 3 components (RGB)"
            assert all(isinstance(c, int) for c in color), (
                f"Color {i} should have integer components"
            )
            assert all(0 <= c <= 255 for c in color), (
                f"Color {i} components should be in range [0, 255]"
            )

    def test_mouse_colors_first_color_is_red(self):
        """Test that the first mouse color is red."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        assert config.MOUSE_COLORS[0] == (228, 26, 28), (
            "First color should be red (228, 26, 28)"
        )

    def test_mouse_colors_are_unique(self):
        """Test that all mouse colors are unique."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        assert len(config.MOUSE_COLORS) == len(set(config.MOUSE_COLORS)), (
            "All mouse colors should be unique"
        )


class TestPoseUtilsConfigStaticObjects:
    """Test static object configuration."""

    def test_static_obj_xy_structure(self):
        """Test that STATIC_OBJ_XY has expected structure."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        assert isinstance(config.STATIC_OBJ_XY, dict), "STATIC_OBJ_XY should be a dict"
        assert len(config.STATIC_OBJ_XY) == 3, "STATIC_OBJ_XY should have 3 entries"

    def test_static_obj_xy_keys(self):
        """Test that STATIC_OBJ_XY has expected keys."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        expected_keys = {"lixit", "food_hopper", "corners"}
        assert set(config.STATIC_OBJ_XY.keys()) == expected_keys, (
            f"STATIC_OBJ_XY should have keys: {expected_keys}"
        )

    def test_static_obj_xy_values(self):
        """Test that STATIC_OBJ_XY has expected boolean values."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        assert config.STATIC_OBJ_XY["lixit"] is False
        assert config.STATIC_OBJ_XY["food_hopper"] is False
        assert config.STATIC_OBJ_XY["corners"] is True

        # All values should be booleans
        for key, value in config.STATIC_OBJ_XY.items():
            assert isinstance(value, bool), f"Value for {key} should be a boolean"

    def test_static_obj_colors_structure(self):
        """Test that STATIC_OBJ_COLORS has expected structure."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        assert isinstance(config.STATIC_OBJ_COLORS, dict), (
            "STATIC_OBJ_COLORS should be a dict"
        )
        assert len(config.STATIC_OBJ_COLORS) == 3, (
            "STATIC_OBJ_COLORS should have 3 entries"
        )

    def test_static_obj_colors_keys(self):
        """Test that STATIC_OBJ_COLORS has expected keys."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        expected_keys = {"lixit", "food_hopper", "corners"}
        assert set(config.STATIC_OBJ_COLORS.keys()) == expected_keys, (
            f"STATIC_OBJ_COLORS should have keys: {expected_keys}"
        )

    def test_static_obj_colors_values_are_valid_rgb(self):
        """Test that all static object colors are valid RGB tuples."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        for obj_name, color in config.STATIC_OBJ_COLORS.items():
            assert isinstance(color, tuple), f"Color for {obj_name} should be a tuple"
            assert len(color) == 3, (
                f"Color for {obj_name} should have 3 components (RGB)"
            )
            assert all(isinstance(c, int) for c in color), (
                f"Color for {obj_name} should have integer components"
            )
            assert all(0 <= c <= 255 for c in color), (
                f"Color for {obj_name} components should be in range [0, 255]"
            )

    def test_static_obj_colors_specific_values(self):
        """Test that static object colors have expected specific values."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        assert config.STATIC_OBJ_COLORS["lixit"] == (55, 126, 184), (
            "Lixit color should be blue (55, 126, 184)"
        )
        assert config.STATIC_OBJ_COLORS["food_hopper"] == (255, 127, 0), (
            "Food hopper color should be orange (255, 127, 0)"
        )
        assert config.STATIC_OBJ_COLORS["corners"] == (75, 175, 74), (
            "Corners color should be green (75, 175, 74)"
        )

    def test_static_obj_xy_and_colors_have_same_keys(self):
        """Test that STATIC_OBJ_XY and STATIC_OBJ_COLORS have matching keys."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        assert set(config.STATIC_OBJ_XY.keys()) == set(
            config.STATIC_OBJ_COLORS.keys()
        ), "STATIC_OBJ_XY and STATIC_OBJ_COLORS should have the same keys"


class TestPoseUtilsConfigInstantiation:
    """Test PoseUtilsConfig instantiation and Pydantic behavior."""

    def test_can_instantiate_without_arguments(self):
        """Test that config can be instantiated without any arguments."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        assert config is not None
        assert isinstance(config, PoseUtilsConfig)

    def test_multiple_instances_have_same_defaults(self):
        """Test that multiple instances have identical default values."""
        # Arrange & Act
        config1 = PoseUtilsConfig()
        config2 = PoseUtilsConfig()

        # Assert
        assert config1.NOSE_INDEX == config2.NOSE_INDEX
        assert config1.MIN_HIGH_CONFIDENCE == config2.MIN_HIGH_CONFIDENCE
        assert config1.MOUSE_COLORS == config2.MOUSE_COLORS
        assert config1.STATIC_OBJ_XY == config2.STATIC_OBJ_XY
        assert config1.CONNECTED_SEGMENTS == config2.CONNECTED_SEGMENTS

    def test_config_is_independent(self):
        """Test that modifying one instance doesn't affect another."""
        # Arrange
        config1 = PoseUtilsConfig()
        config2 = PoseUtilsConfig()

        # Act
        config1.MIN_HIGH_CONFIDENCE = 0.9

        # Assert
        assert config1.MIN_HIGH_CONFIDENCE == 0.9
        assert config2.MIN_HIGH_CONFIDENCE == 0.75, (
            "Modifying config1 should not affect config2"
        )


class TestPoseUtilsConfigIntegrity:
    """Test overall integrity and consistency of the configuration."""

    def test_all_keypoint_indices_used_in_connected_segments(self):
        """Test that most keypoint indices appear in CONNECTED_SEGMENTS."""
        # Arrange & Act
        config = PoseUtilsConfig()

        all_segment_indices = set()
        for segment in config.CONNECTED_SEGMENTS:
            all_segment_indices.update(segment)

        # Assert
        # Not all keypoints need to be in segments (ears are typically not connected)
        # but the main body structure should be present
        assert config.NOSE_INDEX in all_segment_indices
        assert config.BASE_NECK_INDEX in all_segment_indices
        assert config.CENTER_SPINE_INDEX in all_segment_indices
        assert config.BASE_TAIL_INDEX in all_segment_indices
        assert config.MID_TAIL_INDEX in all_segment_indices
        assert config.TIP_TAIL_INDEX in all_segment_indices

    def test_confidence_thresholds_ordered(self):
        """Test that high confidence threshold is higher than task-specific thresholds."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        assert config.MIN_HIGH_CONFIDENCE >= config.MIN_GAIT_CONFIDENCE, (
            "MIN_HIGH_CONFIDENCE should be >= MIN_GAIT_CONFIDENCE"
        )
        assert config.MIN_HIGH_CONFIDENCE >= config.MIN_JABS_CONFIDENCE, (
            "MIN_HIGH_CONFIDENCE should be >= MIN_JABS_CONFIDENCE"
        )

    def test_min_jabs_keypoints_reasonable(self):
        """Test that MIN_JABS_KEYPOINTS is reasonable for 12 keypoint system."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        assert 1 <= config.MIN_JABS_KEYPOINTS <= 12, (
            "MIN_JABS_KEYPOINTS should be between 1 and 12"
        )

    def test_enough_mouse_colors_for_typical_cages(self):
        """Test that there are enough colors for typical multi-mouse cages."""
        # Arrange & Act
        config = PoseUtilsConfig()

        # Assert
        # Typical cages have up to 4-5 mice, but having 11 colors provides good coverage
        assert len(config.MOUSE_COLORS) >= 4, (
            "Should have at least 4 colors for typical multi-mouse scenarios"
        )
