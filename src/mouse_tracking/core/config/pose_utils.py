from pydantic_settings import BaseSettings


class PoseUtilsConfig(BaseSettings):
    """Configuration for pose utility functions."""

    NOSE_INDEX: int = 0
    LEFT_EAR_INDEX: int = 1
    RIGHT_EAR_INDEX: int = 2
    BASE_NECK_INDEX: int = 3
    LEFT_FRONT_PAW_INDEX: int = 4
    RIGHT_FRONT_PAW_INDEX: int = 5
    CENTER_SPINE_INDEX: int = 6
    LEFT_REAR_PAW_INDEX: int = 7
    RIGHT_REAR_PAW_INDEX: int = 8
    BASE_TAIL_INDEX: int = 9
    MID_TAIL_INDEX: int = 10
    TIP_TAIL_INDEX: int = 11

    CONNECTED_SEGMENTS: list[list[int]] = [
        [LEFT_FRONT_PAW_INDEX, CENTER_SPINE_INDEX, RIGHT_FRONT_PAW_INDEX],
        [LEFT_REAR_PAW_INDEX, BASE_TAIL_INDEX, RIGHT_REAR_PAW_INDEX],
        [
            NOSE_INDEX,
            BASE_NECK_INDEX,
            CENTER_SPINE_INDEX,
            BASE_TAIL_INDEX,
            MID_TAIL_INDEX,
            TIP_TAIL_INDEX,
        ],
    ]

    MIN_HIGH_CONFIDENCE: float = 0.75
    MIN_GAIT_CONFIDENCE: float = 0.3
    MIN_JABS_CONFIDENCE: float = 0.3
    MIN_JABS_KEYPOINTS: int = 3

    # Large animals are rarely larger than 100px in our OFA
    OFA_MAX_EXPECTED_AREA_PX: int = 150 * 150

    # Colors
    MOUSE_COLORS: list[tuple[int, int, int]] = [
        (228, 26, 28),  # Red
        (152, 78, 163),  # Purple
        (255, 255, 51),  # Yellow
        (166, 86, 40),  # Brown
        (247, 129, 191),  # Pink
        (166, 206, 227),  # Light Blue
        (178, 223, 138),  # Light Green
        (251, 154, 153),  # Peach
        (253, 191, 111),  # Light Orange
        (202, 178, 214),  # Light Purple
        (255, 255, 153),  # Faded Yellow
    ]

    # Static object settings
    STATIC_OBJ_XY: dict[str, bool] = {
        "lixit": False,
        "food_hopper": False,
        "corners": True,
    }
    STATIC_OBJ_COLORS: dict[str, tuple[int, int, int]] = {
        "lixit": (55, 126, 184),  # Water spout is Blue
        "food_hopper": (255, 127, 0),  # Food hopper is Orange
        "corners": (75, 175, 74),  # Arena corners are Green
    }
