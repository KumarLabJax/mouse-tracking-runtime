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

from .core import (
    Detection,
    Tracklet,
    Fragment,
    VideoObservations,
    get_point_dist,
    compare_pose_and_contours,
    make_pose_seg_dist_mat,
    hungarian_match_points_seg,
)

from .vectorized_features import (
    VectorizedDetectionFeatures,
    compute_vectorized_pose_distances,
    compute_vectorized_embedding_distances,
    compute_vectorized_segmentation_ious,
    compute_vectorized_match_costs,
)

from .greedy_matching import vectorized_greedy_matching

from .batch_processing import BatchedFrameProcessor

__all__ = [
    # Core classes
    "Detection",
    "Tracklet", 
    "Fragment",
    "VideoObservations",
    
    # Core functions
    "get_point_dist",
    "compare_pose_and_contours",
    "make_pose_seg_dist_mat",
    "hungarian_match_points_seg",
    
    # Vectorized features
    "VectorizedDetectionFeatures",
    "compute_vectorized_pose_distances",
    "compute_vectorized_embedding_distances",
    "compute_vectorized_segmentation_ious",
    "compute_vectorized_match_costs",
    
    # Optimized algorithms
    "vectorized_greedy_matching",
    "BatchedFrameProcessor",
]