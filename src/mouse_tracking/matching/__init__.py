"""Mouse tracking matching module.

This module provides efficient algorithms for matching detections across video frames
and building tracklets from pose estimation and segmentation data.

Main components:
- Detection: Individual detection with pose, embedding, and segmentation data
- Tracklet: Sequence of linked detections across frames
- Fragment: Collection of overlapping tracklets
- VideoObservations: Main orchestration class for video processing

Key algorithms:
- Vectorized distance computation for efficient batch processing
- Optimized O(k log k) greedy matching algorithm
- Memory-efficient batch processing for large videos
- Tracklet stitching for long-term identity management
"""

from .batch_processing import BatchedFrameProcessor
from .core import (
    Fragment,
    Tracklet,
    VideoObservations,
    compare_pose_and_contours,
    get_point_dist,
    hungarian_match_points_seg,
    make_pose_seg_dist_mat,
)
from .detection import Detection
from .greedy_matching import vectorized_greedy_matching
from .vectorized_features import (
    VectorizedDetectionFeatures,
    compute_vectorized_embedding_distances,
    compute_vectorized_match_costs,
    compute_vectorized_pose_distances,
    compute_vectorized_segmentation_ious,
)

__all__ = [
    "BatchedFrameProcessor",
    "Detection",
    "Fragment",
    "Tracklet",
    "VectorizedDetectionFeatures",
    "VideoObservations",
    "compare_pose_and_contours",
    "compute_vectorized_embedding_distances",
    "compute_vectorized_match_costs",
    "compute_vectorized_pose_distances",
    "compute_vectorized_segmentation_ious",
    "get_point_dist",
    "hungarian_match_points_seg",
    "make_pose_seg_dist_mat",
    "vectorized_greedy_matching",
]
