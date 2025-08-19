import cv2
import numpy as np

from mouse_tracking.core.exceptions import InvalidIdentityException


def get_rotation_mat(
    pose: np.ndarray, input_size: tuple[int], output_size: tuple[int]
) -> np.ndarray:
    """Generates a rotation matrix based on a pose.

    Args:
            pose: pose data align (sorted [y, x])
            input_size: input image size [l, w]
            output_size: output image size [l, w]

    Returns:
            transformation matrix of shape [2, 3].
            When used with `cv2.warpAffine`, will crop and rotate such that the pose nose point is aligned to the 0 direction (pointing right).

    Raises:
            InvalidIdentityException when the pose cannot be used to generate a cropped input.

    Notes:
            The final transformation matrix is a combination of 3 transformations:
            1. Translation of mouse to center coordinate system
            2. Rotation of mouse to point right
            3. Translation of mouse to center of output
    """
    masked_pose = np.ma.array(
        np.flip(pose, axis=-1),
        mask=np.repeat(np.all(pose == 0, axis=-1), 2).reshape(pose.shape),
    )
    if np.all(masked_pose.mask[0:10]):
        raise InvalidIdentityException(
            "Pose required at least 1 keypoint on the main torso to crop and rotate frame."
        )
    if np.all(masked_pose.mask[0:4]):
        raise InvalidIdentityException(
            "Pose required at least 1 keypoint on the front to crop and rotate frame."
        )
    # Use all non-tail keypoints for center of crop
    center = (
        (np.max(masked_pose[0:10], axis=0) + np.min(masked_pose[0:10], axis=0)) / 2
    ).filled()
    # Use the face keypoints for center direction
    center_face = (
        (np.max(masked_pose[0:4], axis=0) + np.min(masked_pose[0:4], axis=0)) / 2
    ).filled()
    distance = center_face - center
    norm = np.hypot(distance[0], distance[1])
    rot_cos = distance[0] / norm  # cos(-θ) = cos(θ)
    rot_sin = -distance[1] / norm  # sin(-θ) = -sin(θ)
    translate_1 = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])
    rotate = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
    translate_2 = np.array(
        [[1, 0, output_size[0] / 2], [0, 1, output_size[1] / 2], [0, 0, 1]]
    )
    aff_mat = np.matmul(np.matmul(translate_2, rotate), translate_1)
    return aff_mat[:2]


def crop_and_rotate_frame(
    frame: np.ndarray, pose: np.ndarray, crop_size: tuple[int]
) -> np.ndarray:
    """Crops and rotates a frame based on pose predictions.

    Args:
            frame: frame to crop and rotate
            pose: pose to use in transformation (sorted [y, x])
            crop_size: size of the resulting cropped frame

    Returns:
            cropped and rotated frame.
            Mouse's nose will be pointing left.
    """
    warped_frame = np.copy(frame)
    aff_mat = get_rotation_mat(pose, frame.shape[:2], crop_size)
    warped_frame = cv2.warpAffine(warped_frame, aff_mat, (128, 128))
    # Right now, the frame is nose pointing right, so rotate it 180 deg because the model trains on "pointing left" (the tensorflow 0 direction)
    warped_frame = cv2.rotate(warped_frame, cv2.ROTATE_180)
    return warped_frame
