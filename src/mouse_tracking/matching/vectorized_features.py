"""Vectorized feature extraction and distance computation for mouse tracking."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.spatial.distance

from mouse_tracking.matching.detection import Detection
from mouse_tracking.utils.segmentation import render_blob


class VectorizedDetectionFeatures:
    """Precomputed vectorized features for batch detection processing."""

    def __init__(self, detections: list[Detection]):
        """Initialize vectorized features from a list of detections.

        Args:
                detections: List of Detection objects to extract features from
        """
        self.n_detections = len(detections)
        self.detections = detections

        # Extract and organize features into arrays
        self.poses = self._extract_poses(detections)  # Shape: (n, 12, 2)
        self.embeddings = self._extract_embeddings(detections)  # Shape: (n, embed_dim)
        self.valid_pose_masks = self._compute_valid_pose_masks()  # Shape: (n, 12)
        self.valid_embed_masks = self._compute_valid_embed_masks()  # Shape: (n,)

        # Cache rotated poses for efficiency
        self._rotated_poses = None
        self._seg_images = None

    def _extract_poses(self, detections: list[Detection]) -> np.ndarray:
        """Extract pose data into a vectorized array."""
        poses = []
        for det in detections:
            if det.pose is not None:
                poses.append(det.pose)
            else:
                # Default to zeros for missing poses
                poses.append(np.zeros((12, 2), dtype=np.float64))
        return np.array(poses, dtype=np.float64)

    def _extract_embeddings(self, detections: list[Detection]) -> np.ndarray:
        """Extract embedding data into a vectorized array."""
        embeddings = []
        embed_dim = None

        # First pass: determine embedding dimension from any non-None embedding
        for det in detections:
            if det.embed is not None:
                embed_dim = len(det.embed)
                break

        if embed_dim is None:
            # No embeddings found at all, return empty array
            return np.array([]).reshape(self.n_detections, 0)

        # Second pass: extract embeddings, preserving zeros as they are used for invalid detection
        for det in detections:
            if det.embed is not None and len(det.embed) == embed_dim:
                embeddings.append(det.embed)
            else:
                # Default to zeros for missing embeddings
                embeddings.append(np.zeros(embed_dim, dtype=np.float64))

        return np.array(embeddings, dtype=np.float64)

    def _compute_valid_pose_masks(self) -> np.ndarray:
        """Compute valid keypoint masks for all poses."""
        # Valid keypoints are those that are not all zeros
        return ~np.all(self.poses == 0, axis=-1)  # Shape: (n, 12)

    def _compute_valid_embed_masks(self) -> np.ndarray:
        """Compute valid embedding masks."""
        if self.embeddings.size == 0:
            return np.zeros(self.n_detections, dtype=bool)
        return ~np.all(self.embeddings == 0, axis=-1)  # Shape: (n,)

    def get_rotated_poses(self) -> np.ndarray:
        """Get 180-degree rotated poses for all detections."""
        if self._rotated_poses is not None:
            return self._rotated_poses

        rotated_poses = np.zeros_like(self.poses)

        # Import Detection here to avoid circular imports
        from mouse_tracking.matching.core import Detection

        for i, det in enumerate(self.detections):
            if det.pose is not None:
                # Use the existing rotate_pose method but cache result
                rotated_poses[i] = Detection.rotate_pose(det.pose, 180)
            else:
                rotated_poses[i] = self.poses[i]  # zeros

        self._rotated_poses = rotated_poses
        return self._rotated_poses

    def get_seg_images(self) -> list[np.ndarray]:
        """Get segmentation images for all detections."""
        if self._seg_images is not None:
            return self._seg_images

        seg_images = []
        for det in self.detections:
            if det._seg_mat is not None:
                seg_images.append(render_blob(det._seg_mat))
            else:
                seg_images.append(None)

        self._seg_images = seg_images
        return self._seg_images


def compute_vectorized_pose_distances(
    features1: VectorizedDetectionFeatures,
    features2: VectorizedDetectionFeatures,
    use_rotation: bool = False,
) -> np.ndarray:
    """Compute pose distance matrix between two sets of detection features.

    Args:
            features1: First set of detection features
            features2: Second set of detection features
            use_rotation: Whether to consider 180-degree rotated poses

    Returns:
            Distance matrix of shape (n1, n2) with mean pose distances
    """
    poses1 = features1.poses  # Shape: (n1, 12, 2)
    poses2 = features2.poses  # Shape: (n2, 12, 2)
    valid1 = features1.valid_pose_masks  # Shape: (n1, 12)
    valid2 = features2.valid_pose_masks  # Shape: (n2, 12)

    # Broadcasting: (n1, 1, 12, 2) - (1, n2, 12, 2) = (n1, n2, 12, 2)
    diff = poses1[:, None, :, :] - poses2[None, :, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))  # (n1, n2, 12)

    # Vectorized valid comparison mask: (n1, 1, 12) & (1, n2, 12) = (n1, n2, 12)
    valid_comparisons = valid1[:, None, :] & valid2[None, :, :]

    # Compute mean distances where valid comparisons exist
    result = np.full((features1.n_detections, features2.n_detections), np.nan)

    # For each pair, check if any valid comparisons exist
    any_valid = np.any(valid_comparisons, axis=-1)  # (n1, n2)

    # Compute mean distances only where valid comparisons exist
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_distances = np.where(
            any_valid, np.mean(distances, axis=-1, where=valid_comparisons), np.nan
        )

    if use_rotation:
        # Also compute distances with rotated poses
        rotated_poses1 = features1.get_rotated_poses()

        # Recompute with rotated poses1
        diff_rot = rotated_poses1[:, None, :, :] - poses2[None, :, :, :]
        distances_rot = np.sqrt(np.sum(diff_rot**2, axis=-1))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_distances_rot = np.where(
                any_valid,
                np.mean(distances_rot, axis=-1, where=valid_comparisons),
                np.nan,
            )

        # Take minimum of regular and rotated distances
        result = np.where(
            np.isnan(mean_distances),
            mean_distances_rot,
            np.where(
                np.isnan(mean_distances_rot),
                mean_distances,
                np.minimum(mean_distances, mean_distances_rot),
            ),
        )
    else:
        result = mean_distances

    return result


def compute_vectorized_embedding_distances(
    features1: VectorizedDetectionFeatures, features2: VectorizedDetectionFeatures
) -> np.ndarray:
    """Compute embedding distance matrix between two sets of detection features.

    Args:
            features1: First set of detection features
            features2: Second set of detection features

    Returns:
            Distance matrix of shape (n1, n2) with cosine distances
    """
    if features1.embeddings.size == 0 or features2.embeddings.size == 0:
        return np.full((features1.n_detections, features2.n_detections), np.nan)

    valid1 = features1.valid_embed_masks
    valid2 = features2.valid_embed_masks

    # Extract valid embeddings only
    valid_embeds1 = features1.embeddings[valid1]
    valid_embeds2 = features2.embeddings[valid2]

    if len(valid_embeds1) == 0 or len(valid_embeds2) == 0:
        return np.full((features1.n_detections, features2.n_detections), np.nan)

    # Compute cosine distances using scipy
    valid_distances = scipy.spatial.distance.cdist(
        valid_embeds1, valid_embeds2, metric="cosine"
    )
    valid_distances = np.clip(valid_distances, 0, 1.0 - 1e-8)

    # Map back to full matrix
    result = np.full((features1.n_detections, features2.n_detections), np.nan)
    valid1_indices = np.where(valid1)[0]
    valid2_indices = np.where(valid2)[0]

    for i, idx1 in enumerate(valid1_indices):
        for j, idx2 in enumerate(valid2_indices):
            result[idx1, idx2] = valid_distances[i, j]

    return result


def compute_vectorized_segmentation_ious(
    features1: VectorizedDetectionFeatures, features2: VectorizedDetectionFeatures
) -> np.ndarray:
    """Compute segmentation IoU matrix between two sets of detection features.

    Args:
            features1: First set of detection features
            features2: Second set of detection features

    Returns:
            IoU matrix of shape (n1, n2) with intersection over union values
    """
    seg_images1 = features1.get_seg_images()
    seg_images2 = features2.get_seg_images()

    result = np.full((features1.n_detections, features2.n_detections), np.nan)

    for i, seg1 in enumerate(seg_images1):
        for j, seg2 in enumerate(seg_images2):
            # Handle cases where segmentations exist (even if rendered as all zeros)
            # This matches the original Detection.seg_iou behavior
            if seg1 is not None and seg2 is not None:
                # Compute IoU using the same logic as Detection.seg_iou
                intersection = np.sum(np.logical_and(seg1, seg2))
                union = np.sum(np.logical_or(seg1, seg2))
                if union == 0:
                    result[i, j] = 0.0
                else:
                    result[i, j] = intersection / union
            elif (
                features1.detections[i]._seg_mat is not None
                or features2.detections[j]._seg_mat is not None
            ):
                # If at least one has segmentation data (even if rendered as zeros), return 0.0
                # This matches the original behavior where render_blob creates an image
                result[i, j] = 0.0
            # else remains NaN for cases where both segmentations are truly missing

    return result


def compute_vectorized_match_costs(
    features1: VectorizedDetectionFeatures,
    features2: VectorizedDetectionFeatures,
    max_dist: float = 40,
    default_cost: float | tuple[float] = 0.0,
    beta: tuple[float] = (1.0, 1.0, 1.0),
    pose_rotation: bool = False,
) -> np.ndarray:
    """Compute full match cost matrix between two sets of detection features.

    This vectorized version replicates the logic of Detection.calculate_match_cost
    but computes all pairwise costs in batches for better performance.

    Args:
            features1: First set of detection features
            features2: Second set of detection features
            max_dist: Distance at which maximum penalty is applied for poses
            default_cost: Default cost for missing data (pose, embed, seg)
            beta: Scaling factors for (pose, embed, seg) costs
            pose_rotation: Whether to consider 180-degree rotated poses

    Returns:
            Cost matrix of shape (n1, n2) with match costs
    """
    assert len(beta) == 3
    assert isinstance(default_cost, float | int) or len(default_cost) == 3

    if isinstance(default_cost, float | int):
        default_pose_cost = default_cost
        default_embed_cost = default_cost
        default_seg_cost = default_cost
    else:
        default_pose_cost, default_embed_cost, default_seg_cost = default_cost

    n1, n2 = features1.n_detections, features2.n_detections

    # Compute all distance matrices
    pose_distances = compute_vectorized_pose_distances(
        features1, features2, use_rotation=pose_rotation
    )
    embed_distances = compute_vectorized_embedding_distances(features1, features2)
    seg_ious = compute_vectorized_segmentation_ious(features1, features2)

    # Convert distances to costs using the same logic as the original method

    # Pose costs
    pose_costs = np.full((n1, n2), np.log(1e-8) * default_pose_cost)
    valid_pose = ~np.isnan(pose_distances)
    pose_costs[valid_pose] = np.log(
        (1 - np.clip(pose_distances[valid_pose] / max_dist, 0, 1)) + 1e-8
    )

    # Embedding costs
    embed_costs = np.full((n1, n2), np.log(1e-8) * default_embed_cost)
    valid_embed = ~np.isnan(embed_distances)
    embed_costs[valid_embed] = np.log((1 - embed_distances[valid_embed]) + 1e-8)

    # Segmentation costs
    seg_costs = np.full((n1, n2), np.log(1e-8) * default_seg_cost)
    valid_seg = ~np.isnan(seg_ious)
    seg_costs[valid_seg] = np.log(seg_ious[valid_seg] + 1e-8)

    # Combine costs using beta weights
    final_costs = -(
        pose_costs * beta[0] + embed_costs * beta[1] + seg_costs * beta[2]
    ) / np.sum(beta)

    return final_costs
