"""Module for definition of the Detection class."""

import h5py
import numpy as np
import scipy

from mouse_tracking.utils.segmentation import render_blob


class Detection:
    """Detection object that describes a linked pose and segmentation."""

    def __init__(
        self,
        frame: int | None = None,
        pose_idx: int | None = None,
        pose: np.ndarray = None,
        embed: np.ndarray = None,
        seg_idx: int | None = None,
        seg: np.ndarray = None,
    ) -> None:
        """Initializes a detection object from observation data.

        Args:
                frame: index describing the frame where the observation exists
                pose_idx: pose index in the pose file
                pose: numpy array of [12, 2] containing pose data
                embed: vector of arbitrary length containing embedding data
                seg_idx: segmentation index in the pose file
                seg: a full matrix of segmentation data (-1 padded)
        """
        # Information about how this detection was produced.
        self._frame = frame
        self._pose_idx = pose_idx
        self._seg_idx = seg_idx
        # Information about this detection for matching with other detections.
        self._pose = pose
        self._embed = embed
        self._seg_mat = seg
        self._cached = False
        self._seg_img = None

    @classmethod
    def from_pose_file(cls, pose_file, frame, pose_idx, seg_idx):
        """Initializes a detection from a given pose file.

        Args:
                pose_file: input pose file
                frame: frame index where the pose is present
                pose_idx: pose index
                seg_idx: segmentation index

        Notes:
                This is for convenience for smaller tests. Using h5py to read chunks this small is very inefficient for large files.
        """
        with h5py.File(pose_file, "r") as f:
            if pose_idx is not None:
                pose = f["poseest/points"][frame, pose_idx]
                embed = f["poseest/identity_embeds"][frame, pose_idx]
            else:
                pose = None
                embed = None
            seg = f["poseest/seg_data"][frame, seg_idx] if seg_idx is not None else None
        return cls(frame, pose_idx, pose, embed, seg_idx, seg)

    @staticmethod
    def pose_distance(points_1, points_2) -> float:
        """Calculates the mean distance between all keypoits.

        Args:
                points_1: first set of keypoints of shape [n_keypoints, 2]
                points_2: second set of keypoints of shape [n_keypoints, 2]

        Returns:
                mean distance between all valid keypoints
        """
        if points_1 is None or points_2 is None:
            return np.nan
        p1_valid = ~np.all(points_1 == 0, axis=-1)
        p2_valid = ~np.all(points_2 == 0, axis=-1)
        valid_comparisons = np.logical_and(p1_valid, p2_valid)
        # no overlapping keypoints
        if np.all(~valid_comparisons):
            return np.nan
        diff = points_1.astype(np.float64) - points_2.astype(np.float64)
        dists = np.hypot(diff[:, 0], diff[:, 1])
        return np.mean(dists, where=valid_comparisons)

    @staticmethod
    def rotate_pose(
        points: np.ndarray, angle: float, center: np.ndarray = None
    ) -> np.ndarray:
        """Rotates a pose around its center by an angle.

        Args:
                points: keypoint data of shape [n_keypoints, 2]
                angle: angle in degrees to rotate
                center: optional center of rotation. If not provided, the mean of non-tail keypoints are used as the center.

        Returns:
                rotated keypoints
        """
        points_valid = ~np.all(points == 0, axis=-1)
        # No points to rotate, just return original points.
        if np.all(~points_valid):
            return points
        if center is None:
            # Can't calculate a center to rotate only tail keypoints, just return them
            if np.all(~points_valid[:10]):
                return points
            center = np.mean(
                points[:10],
                axis=0,
                where=np.repeat(points_valid[:, np.newaxis], 2, 1)[:10],
            )
        angle_rad = np.deg2rad(angle)
        R = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )
        o = np.atleast_2d(center)
        p = np.atleast_2d(points)
        rotated_pose = np.squeeze((R @ (p.T - o.T) + o.T).T)
        rotated_pose[~points_valid] = 0
        return rotated_pose

    @staticmethod
    def embed_distance(embed_1, embed_2) -> float:
        """Calculates the cosine distance between two embeddings.

        Args:
                embed_1: first embedded vector
                embed_2: second embedded vector

        Returns:
                cosine distance between the embeddings
        """
        # Check for default embeddings
        if np.all(embed_1 == 0) or np.all(embed_2 == 0):
            return np.nan
        return np.clip(
            scipy.spatial.distance.cdist([embed_1], [embed_2], metric="cosine")[0][0],
            0,
            1.0 - 1e-8,
        )

    @staticmethod
    def seg_iou(seg_1, seg_2) -> float:
        """Calculates the IoU for a pair of segmentations.

        Args:
                seg_1: padded contour data for the first segmentation
                seg_2: padded contour data for the second segmentation

        Returns:
                IoU between segmentations
        """
        intersection = np.sum(np.logical_and(seg_1, seg_2))
        union = np.sum(np.logical_or(seg_1, seg_2))
        # division by 0 safety
        if union == 0:
            return 0.0
        else:
            return intersection / union

    @staticmethod
    def calculate_match_cost_multi(args):
        """Thin wrapper for `calculate_match_cost` with a single arg for working with multiprocessing library."""
        (detection_1, detection_2, max_dist, default_cost, beta, pose_rotation) = args
        return Detection.calculate_match_cost(
            detection_1, detection_2, max_dist, default_cost, beta, pose_rotation
        )

    @staticmethod
    def calculate_match_cost(
        detection_1: "Detection",
        detection_2: "Detection",
        max_dist: float = 40,
        default_cost: float | tuple[float] = 0.0,
        beta: tuple[float] = (1.0, 1.0, 1.0),
        pose_rotation: bool = False,
    ) -> float:
        """Defines the matching cost between detections.

        Args:
                detection_1: Detection to compare
                detection_2: Detection to compare
                max_dist: distance at which maximum penalty is applied
                default_cost: Float or Tuple of length 3 containing the default cost for linking (pose, embed, segmentation). Default value is used when either observation cannot be compared. Should be range 0-1 (min-max penalty).
                beta: Tuple of length 3 containing the scaling factors for costs. Scaling calculated via sigma(beta*cost)/sigma(beta) to preserve scale. Supplying values of (1,0,0) would indicate only using pose matching.
                pose_rotation: Allow the pose to be rotated by 180 deg for distance calculation. Our pose model sometimes has trouble predicting the correct nose/tail. This allows 180deg rotations between frames to not be penalized for matching.

        Returns:
                -log probability of the 2 detections getting linked

        We scale all the values between 0-1, then apply a log (with 1e-8 added)
        This results in a cost range per-value of 0 to -18.42
        """
        assert len(beta) == 3
        assert isinstance(default_cost, float | int) == 1 or len(default_cost) == 3

        if isinstance(default_cost, float | int):
            default_pose_cost = default_cost
            default_embed_cost = default_cost
            default_seg_cost = default_cost
        else:
            default_pose_cost, default_embed_cost, default_seg_cost = default_cost

        # Pose link cost
        pose_dist = Detection.pose_distance(detection_1.pose, detection_2.pose)
        if pose_rotation:
            # While we might get a slightly different result if we do all combinations of rotations, we skip those for efficiency
            alt_pose_dist = Detection.pose_distance(
                detection_1.get_rotated_pose(), detection_2.pose
            )
            if alt_pose_dist < pose_dist:
                pose_dist = alt_pose_dist
        if not np.isnan(pose_dist):
            # max_dist pixel or greater distance gets a maximum cost
            pose_cost = np.log((1 - np.clip(pose_dist / max_dist, 0, 1)) + 1e-8)
        else:
            pose_cost = np.log(1e-8) * default_pose_cost
        # Our ReID network operates on a cosine distance, which is already scaled from 0-1
        embed_dist = Detection.embed_distance(detection_1.embed, detection_2.embed)
        if not np.isnan(embed_dist):
            embed_cost = np.log((1 - embed_dist) + 1e-8)
            # Publication cost for ReID net here:
            # embed_cost = stats.multivariate_normal.logpdf(detection_1.embed, mean=detection_2.embed, cov=np.diag(np.repeat(10**2, len(detection_1.embed)))) / 5
        else:
            # Penalty for no embedding (probably bad pose)
            embed_cost = np.log(1e-8) * default_embed_cost
        # Segmentation link cost
        seg_dist = Detection.seg_iou(detection_1.seg_img, detection_2.seg_img)
        if not np.isnan(seg_dist):
            seg_cost = np.log(seg_dist + 1e-8)
        else:
            # Penalty for no segmentation
            seg_cost = np.log(1e-8) * default_seg_cost
        return -(
            pose_cost * beta[0] + embed_cost * beta[1] + seg_cost * beta[2]
        ) / np.sum(beta)

    @property
    def frame(self):
        """Frame where the observation exists."""
        return self._frame

    @property
    def pose_idx(self):
        """Index of pose in the pose file."""
        return self._pose_idx

    @property
    def pose(self):
        """Pose data."""
        return self._pose

    @property
    def embed(self):
        """Embedding data."""
        return self._embed

    @property
    def seg_idx(self):
        """Index of seg in the pose file."""
        return self._seg_idx

    @property
    def seg_mat(self):
        """Raw segmentation data, as a padded point matrix."""
        return self._seg_mat

    @property
    def seg_img(self):
        """Rendered binary mask of segmentation data."""
        if self._cached:
            return self._seg_img
        return render_blob(self._seg_mat)

    def cache(self):
        """Enables the caching of the segmentation image."""
        # skip operations if already cached
        if self._cached:
            return

        self._seg_img = render_blob(self._seg_mat)
        center = (
            np.mean(np.argwhere(self._seg_img), axis=0)
            if self._seg_mat is not None
            else None
        )
        self._rotated_pose = Detection.rotate_pose(self._pose, 180, center)
        self._cached = True

    def get_rotated_pose(self):
        """Returns a 180 deg rotated pose."""
        if self._cached:
            return self._rotated_pose
        center = (
            np.mean(np.argwhere(self._seg_img), axis=0)
            if self._seg_mat is not None
            else None
        )
        return Detection.rotate_pose(self._pose, 180, center)

    def clear_cache(self):
        """Clears the cached data."""
        self._seg_img = None
        self._rotated_pose = None
        self._cached = False
