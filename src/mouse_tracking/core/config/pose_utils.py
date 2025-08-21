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
