import cv2
import h5py
import numpy as np
from scipy.spatial.distance import cdist

ARENA_SIZE_CM = 20.5 * 2.54  # 20.5 inches to cm

DEFAULT_CM_PER_PX = {
    "ltm": ARENA_SIZE_CM / 701,  # 700.570 +/- 10.952 pixels
    "ofa": ARENA_SIZE_CM / 398,  # 397.992 +/- 8.069 pixels
}

ARENA_IMAGING_RESOLUTION = {
    800: "ltm",
    480: "ofa",
}


def plot_keypoints(
    kp: np.ndarray,
    img: np.ndarray,
    color: tuple = (0, 0, 255),
    is_yx: bool = False,
    include_lines: bool = False,
) -> np.ndarray:
    """Plots keypoints on an image.

    Args:
            kp: keypoints of shape [n_keypoints, 2]
            img: image to render the keypoint on
            color: BGR tuple to render the keypoint
            is_yx: are the keypoints formatted y, x instead of x, y?
            include_lines: also render lines between keypoints?

    Returns:
            Copy of image with the keypoints rendered
    """
    img_copy = img.copy()
    kps_ordered = np.flip(kp, axis=-1) if is_yx else kp
    if include_lines and kps_ordered.ndim == 2 and kps_ordered.shape[0] >= 1:
        img_copy = cv2.drawContours(
            img_copy, [kps_ordered.astype(np.int32)], 0, (0, 0, 0), 2, cv2.LINE_AA
        )
        img_copy = cv2.drawContours(
            img_copy, [kps_ordered.astype(np.int32)], 0, color, 1, cv2.LINE_AA
        )
    for _i, kp_data in enumerate(kps_ordered):
        _ = cv2.circle(
            img_copy, (int(kp_data[0]), int(kp_data[1])), 3, (0, 0, 0), -1, cv2.LINE_AA
        )
        _ = cv2.circle(
            img_copy, (int(kp_data[0]), int(kp_data[1])), 2, color, -1, cv2.LINE_AA
        )
    return img_copy


def measure_pair_dists(keypoints: np.ndarray):
    """Measures pairwise distances between all keypoints.

    Args:
            keypoints: keypoints of shape [n_points, 2]

    Returns:
            Distances of shape [n_comparisons]
    """
    dists = cdist(keypoints, keypoints)
    dists = dists[np.nonzero(np.triu(dists))]
    return dists


def filter_square_keypoints(predictions: np.ndarray, tolerance: float = 25.0):
    """Filters raw predictions for a square object.

    Args:
            predictions: raw predictions of shape [n_predictions, 4, 2]
            tolerance: allowed pixel variation

    Returns:
            Proposed actual keypoint locations of shape [4, 2]

    Raises:
            AssertionError if predictions are not the correct shape
            ValueError if predictions fail the tolerance test
    """
    assert len(predictions.shape) == 3

    filtered_predictions = []
    for i in np.arange(len(predictions)):
        dists = measure_pair_dists(predictions[i])
        sorted_dists = np.sort(dists)
        edges, diags = np.split(sorted_dists, [4], axis=0)
        compare_edges = np.concatenate([np.sqrt(np.square(diags) / 2), edges])
        edge_err = np.abs(compare_edges - np.mean(compare_edges))
        if np.all(edge_err < tolerance):
            filtered_predictions.append(predictions[i])

    if len(filtered_predictions) == 0:
        raise ValueError("No predictions were square.")

    return filter_static_keypoints(np.stack(filtered_predictions), tolerance)


def filter_static_keypoints(predictions: np.ndarray, tolerance: float = 25.0):
    """Filters raw predictions for a static object.

    Args:
            predictions: raw predictions of shape [n_predictions, n_keypoints, 2]
            tolerance: allowed pixel variation

    Returns:
            Proposed actual keypoint locations of shape [n_keypoints, 2]

    Raises:
            AssertionError if predictions are not the correct shape
            ValueError if predictions fail the tolerance test
    """
    assert len(predictions.shape) == 3

    keypoint_motion = np.std(predictions, axis=0)
    keypoint_motion = np.hypot(keypoint_motion[:, 0], keypoint_motion[:, 1])

    if np.any(keypoint_motion > tolerance):
        raise ValueError("Predictions are moving!")

    return np.mean(predictions, axis=0)


def get_affine_xform(
    bbox: np.ndarray,
    img_size: tuple[int] = (512, 512),
    warp_size: tuple[int] = (255, 255),
):
    """Obtains an affine transform for reshaping mask predictins.

    Args:
            bbox: bounding box formatted [x1, y1, x2, y2]
            img_size: size of the image the warped image is going to be placed onto
            warp_size: size of the image being warped

    Returns:
            an affine transform matrix, which can be used with cv2.warpAffine to warp an image onto another.
    """
    # Affine transform requires 3 points for projection
    # Since we only have a box, just pick 3 corners
    from_corners = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.float32)
    # bbox is y1, x1, y2, x2
    to_corners = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]]])
    # Here we multiply by the coordinate system scale
    affine_mat = cv2.getAffineTransform(from_corners, to_corners) * [
        [img_size[0] / warp_size[0]],
        [img_size[1] / warp_size[1]],
    ]
    # Adjust the translation
    # Note that since the scale is from 0-1, we can just force the TL corner to be translated
    affine_mat[:, 2] = [bbox[0] * img_size[0], bbox[1] * img_size[1]]
    return affine_mat


def get_rot_rect(mask: np.ndarray):
    """Obtains a rotated rectangle that bounds a segmentation mask.

    Args:
            mask: image data containing the object. Values < 0.5 indicate background while >= 0.5 indicate foreground.

    Returns:
            4 sorted corners describing the object
    """
    contours, heirarchy = cv2.findContours(
        np.uint8(mask > 0.5), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # Only operate on the largest contour, which is usually the first, but use areas to find it
    largest_contour, max_area = None, 0
    for contour in contours:
        cur_area = cv2.contourArea(contour)
        if cur_area > max_area:
            largest_contour = contour
            max_area = cur_area
    corners = cv2.boxPoints(cv2.minAreaRect(largest_contour))
    return sort_corners(corners, mask.shape[:2])


def sort_corners(corners: np.ndarray, img_size: tuple[int]):
    """Sort the corners to be [TL, TR, BR, BL] from the frame the mouses egocentric viewpoint.

    Args:
            corners: corner data to sort of shape [4, 2] sorted [x, y]
            img_size: Size of the image to detect nearest wall

    Notes:
            This reference fram is NOT the same as the imaging reference. Predictions at the bottom will appear rotated by 180deg.
    """
    # Sort the points clockwise
    sorted_corners = sort_points_clockwise(corners)
    # TL corner will be the first of the 2 corners closest to the wall
    dists_to_wall = [
        cv2.pointPolygonTest(
            np.array(
                [[0, 0], [0, img_size[1]], [img_size[0], img_size[1]], [img_size[0], 0]]
            ),
            sorted_corners[i, :],
            measureDist=1,
        )
        for i in np.arange(4)
    ]
    closer_corners = np.where(dists_to_wall < np.mean(dists_to_wall))
    # This is a circular index so first and last needs to be handled differently
    if np.all(closer_corners[0] == [0, 3]):
        sorted_corners = np.roll(sorted_corners, -3, axis=0)
    else:
        sorted_corners = np.roll(sorted_corners, -np.min(closer_corners), axis=0)
    return sorted_corners


def sort_points_clockwise(points):
    """Sorts a list of points to be clockwise relative to the first point.

    Args:
            points: points to sort of shape [n_points, 2]

    Returns:
            points sorted clockwise
    """
    origin_point = np.mean(points, axis=0)
    vectors = points - origin_point
    vec_angles = np.arctan2(vectors[:, 0], vectors[:, 1])
    sorted_points = points[np.argsort(vec_angles)[::-1], :]
    # Roll the points to have the first point still be first
    first_point_idx = np.where(np.all(sorted_points == points[0], axis=1))[0][0]
    return np.roll(sorted_points, -first_point_idx, axis=0)


def get_mask_corners(box: np.ndarray, mask: np.ndarray, img_size: tuple[int]):
    """Finds corners of a mask proposed in a bounding box.

    Args:
            box: bounding box formatted [x1, y1, x2, y2]
            mask: image data containing the object. Values < 0.5 indicate background while >= 0.5 indicate foreground.
            img_size: size of the image where the bounding box resides

    Returns:
            np.ndarray of shape [4, 2] describing the keypoint corners of the box
            See `sort_corner` for order of keypoints.
    """
    affine_mat = get_affine_xform(box, img_size=img_size)
    warped_mask = cv2.warpAffine(mask, affine_mat, (img_size[0], img_size[1]))
    contours, heirarchy = cv2.findContours(
        np.uint8(warped_mask > 0.5), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # Only operate on the largest contour, which is usually the first, but use areas to find it
    largest_contour, max_area = None, 0
    for contour in contours:
        cur_area = cv2.contourArea(contour)
        if cur_area > max_area:
            largest_contour = contour
            max_area = cur_area
    corners = cv2.boxPoints(cv2.minAreaRect(largest_contour))
    return sort_corners(corners, warped_mask.shape[:2])


def get_px_per_cm(corners: np.ndarray, arena_size_cm: float = ARENA_SIZE_CM) -> float:
    """Calculates the pixels per cm conversion for corner predictions.

    Args:
            corners: corner prediction data of shape [4, 2]
            arena_size_cm: size of the arena in cm

    Returns:
            coefficient to multiply pixels to get cm
    """
    dists = measure_pair_dists(corners)
    # Edges are shorter than diagonals
    sorted_dists = np.sort(dists)
    edges = sorted_dists[:4]
    diags = sorted_dists[4:]
    # Calculate all equivalent edge lengths (turn diagonals into equivalent edges)
    edges = np.concatenate([np.sqrt(np.square(diags) / 2), edges])
    cm_per_pixel = np.float32(arena_size_cm / np.mean(edges))

    return cm_per_pixel


def swap_static_obj_xy(pose_file, object_key):
    """Swaps the [y, x] data to [x, y] for a given static object key.

    Args:
            pose_file: pose file to modify in-place
            object_key: dataset key to swap x and y data
    """
    with h5py.File(pose_file, "a") as f:
        if object_key not in f:
            print(f"{object_key} not in {pose_file}.")
            return
        object_data = np.flip(f[object_key][:], axis=-1)
        if len(f[object_key].attrs.keys()) > 0:
            object_attrs = dict(f[object_key].attrs.items())
        else:
            object_attrs = {}
        compression_opt = f[object_key].compression_opts

        del f[object_key]

        if compression_opt is None:
            f.create_dataset(object_key, data=object_data)
        else:
            f.create_dataset(
                object_key,
                data=object_data,
                compression="gzip",
                compression_opts=compression_opt,
            )
        for cur_attr, data in object_attrs.items():
            f[object_key].attrs.create(cur_attr, data)
