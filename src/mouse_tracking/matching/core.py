"""Core matching functions and classes for mouse tracking."""

from __future__ import annotations

import multiprocessing
import warnings
from itertools import chain

import cv2
import h5py
import networkx as nx
import numpy as np
import pandas as pd
import scipy

from mouse_tracking.matching.batch_processing import BatchedFrameProcessor
from mouse_tracking.matching.detection import Detection
from mouse_tracking.matching.greedy_matching import vectorized_greedy_matching
from mouse_tracking.matching.vectorized_features import (
    VectorizedDetectionFeatures,
    compute_vectorized_match_costs,
)
from mouse_tracking.utils.segmentation import get_contour_stack


def get_point_dist(contour: list[np.ndarray], point: np.ndarray):
    """Return the signed distance between a point and a contour.

    Args:
            contour: list of opencv-compliant contours
            point: point of shape [2]

    Returns:
            The largest value "inside" any contour in the list of contours

    Note:
            OpenCV point polygon test defines the signed distance as inside (positive), outside (negative), and on the contour (0).
            Here, we return negative as "inside".
    """
    best_dist = -9999
    for contour_part in contour:
        cur_dist = cv2.pointPolygonTest(contour_part, tuple(point), measureDist=True)
        if cur_dist > best_dist:
            best_dist = cur_dist
    return -best_dist


def compare_pose_and_contours(contours: np.ndarray, poses: np.ndarray):
    """Returns a masked 3D array of signed distances between the pose points and contours.

    Args:
            contours: matrix contour data of shape [n_animals, n_contours, n_points, 2]
            poses: pose data of shape [n_animals, n_keypoints, 2]

    Returns:
            distance matrix between poses and contours of shape [n_valid_poses, n_valid_contours, n_points]

    Notes:
            The shapes are not necessarily the same as the input matrices based on detected default values.
    """
    num_poses = np.sum(~np.all(np.all(poses == 0, axis=2), axis=1))
    num_points = np.shape(poses)[1]
    contour_lists = [
        get_contour_stack(contours[x]) for x in np.arange(np.shape(contours)[0])
    ]
    num_segs = np.count_nonzero(np.array([len(x) for x in contour_lists]))
    if num_poses == 0 or num_segs == 0:
        return None
    dists = np.ma.array(np.zeros([num_poses, num_segs, num_points]), mask=False)
    # TODO: Change this to a vectorized op
    for cur_point in np.arange(num_points):
        for cur_pose in np.arange(num_poses):
            for cur_seg in np.arange(num_segs):
                if np.all(poses[cur_pose, cur_point] == 0):
                    dists.mask[cur_pose, cur_seg, cur_point] = True
                else:
                    dists[cur_pose, cur_seg, cur_point] = get_point_dist(
                        contour_lists[cur_seg], tuple(poses[cur_pose, cur_point])
                    )
    return dists


def make_pose_seg_dist_mat(
    points: np.ndarray,
    seg_contours: np.ndarray,
    ignore_tail: bool = True,
    use_expected_dists: bool = False,
):
    """Helper function to compare poses with contour data.

    Args:
            points: keypoint data for mice of shape [n_animals, n_points, 2] sorted (y, x)
            seg_contours: contour data of shape [n_animals, n_contours, n_points, 2] sorted (x, y)
            ignore_tail: bool to exclude 2 tail keypoints (11 and 12)
            use_expected_dists: adjust distances relative to where the keypoint should be on the mouse

    Returns:
            distance matrix from `compare_pose_and_contours`

    Note: This is a convenience function to run `compare_pose_and_contours` and adjust it more abstractly.
    """
    # Flip the points
    # Also remove the tail points if requested
    if ignore_tail:
        # Remove points 11 and 12, which are mid-tail and tail-tip
        points_mat = np.copy(np.flip(points[:, :11, :], axis=-1))
    else:
        points_mat = np.copy(np.flip(points, axis=-1))
    dists = compare_pose_and_contours(seg_contours, points_mat)
    # Early return if no comparisons were made
    if dists is None:
        return np.ma.array(np.zeros([0, 2], dtype=np.uint32))
    # Suggest matchings based on results
    if not use_expected_dists:
        dists = np.mean(dists, axis=2)
    else:
        # Values of "20" are about midline of an average mouse
        expected_distances = np.array([0, 0, 0, 20, 0, 0, 20, 0, 0, 0, 0, 0])
        # Subtract expected distance
        dists = np.mean(dists - expected_distances[: np.shape(points_mat)[1]], axis=2)
        # Shift to describe "was close to expected"
        dists = -np.abs(dists) + 5
    dists.fill_value = -1
    return dists


def hungarian_match_points_seg(
    points: np.ndarray,
    seg_contours: np.ndarray,
    ignore_tail: bool = True,
    use_expected_dists: bool = False,
    max_dist: float = 0,
):
    """Applies a hungarian matching algorithm to link segs and poses.

    Args:
            points: keypoint data of shape [n_animals, n_points, 2] sorted (y, x)
            seg_contours: padded contour data of shape [n_animals, n_contours, n_points, 2] sorted x, y
            ignore_tail: bool to exclude 2 tail keypoints (11 and 12)
            use_expected_dists: adjust distances relative to where the keypoint should be on the mouse
            max_dist: maximum distance to allow a match. Value of 0 means "average keypoint must be within the segmentation"

    Returns:
            matchings between pose and segmentations of shape [match_idx, 2] where each row is a match between [pose, seg] indices
    """
    dists = make_pose_seg_dist_mat(
        points, seg_contours, ignore_tail, use_expected_dists
    )
    # TODO:
    # Add in filtering out non-unique matches
    hungarian_matches = np.asarray(scipy.optimize.linear_sum_assignment(dists)).T
    filtered_matches = np.array(np.zeros([0, 2], dtype=np.uint32))
    for potential_match in hungarian_matches:
        if dists[potential_match[0], potential_match[1]] < max_dist:
            filtered_matches = np.append(filtered_matches, [potential_match], axis=0)
    return filtered_matches


class Tracklet:
    """An object that stores information about a collection of detections that have been linked together."""

    def __init__(
        self,
        track_id: int | list[int],
        detections: list[Detection],
        additional_embeds: list[np.ndarray] | None = None,
        skip_self_similarity: bool = False,
        embedding_matrix: np.ndarray = None,
    ):
        """Initializes a tracklet object.

        Args:
                track_id: Id of this tracklet. Not used by this class, but holds the value for external applications.
                detections: List of detection objects pertaining to a given tracklet
                additional_embeds: Additional embedding anchors used when calculating distance. Typically these are original tracklet means when tracklets are merged.
                skip_self_similarity: skips the self-similarity calculation and instead just fills with maximal value. Useful for saving on compute.
                embedding_matrix: Overrides embedding matrix. Caution: This is not validated and should only be used for efficiency reasons.
        """
        if additional_embeds is None:
            additional_embeds = []
        self._track_id = track_id if isinstance(track_id, list) else [track_id]
        # Sort the detection frames
        frame_idxs = [x.frame for x in detections if x.frame is not None]
        frame_sort_order = np.argsort(frame_idxs).astype(int).flatten()
        self._detection_list = [detections[x] for x in frame_sort_order]
        self._frames = [frame_idxs[x] for x in frame_sort_order]
        self._start_frame = np.min(self._frames)
        self._end_frame = np.max(self._frames)
        self._n_frames = len(self._frames)
        if embedding_matrix is None:
            self._embeddings = [
                x.embed
                for x in self._detection_list
                if x.embed is not None and np.all(x.embed != 0)
            ]
            if len(self._embeddings) > 0:
                self._embeddings = np.stack(self._embeddings)
        else:
            self._embeddings = embedding_matrix
        self._mean_embed = (
            None if len(self._embeddings) == 0 else np.mean(self._embeddings, axis=0)
        )
        if len(self._embeddings) > 0 and not skip_self_similarity:
            self._median_embed = np.median(self._embeddings, axis=0)
            self._std_embed = np.std(self._embeddings)
            # We can define the confidence we have in the tracklet by looking at the variation in embedding relative to the converged value during the training of the network
            # this value converged to about 0.15, but had variation up to 0.3
            self_similarity = np.clip(
                scipy.spatial.distance.cdist(
                    self._embeddings, [self._mean_embed], metric="cosine"
                ),
                0,
                1.0 - 1e-8,
            )
            self._tracklet_self_similarity = np.mean(self_similarity)
        else:
            self._mean_embed = None
            self._std_embed = None
            self._tracklet_self_similarity = 1.0
        self._additional_embeds = additional_embeds

    @classmethod
    def from_tracklets(
        cls, tracklet_list: list[Tracklet], skip_self_similarity: bool = False
    ):
        """Combines multiple tracklets into one new tracklet.

        Args:
                tracklet_list: list of tracklets to combine
                skip_self_similarity: skips the self-similarity calculation and instead just fills with maximal value. Useful for saving on compute.
        """
        assert len(tracklet_list) > 0
        # track_id can either be an int or a list, so unlist anything
        track_id = list(chain.from_iterable([x.track_id for x in tracklet_list]))
        detections = list(
            chain.from_iterable([x.detection_list for x in tracklet_list])
        )
        mean_embeds = [x.mean_embed for x in tracklet_list]
        extra_embeds = list(
            chain.from_iterable([x.additional_embeds for x in tracklet_list])
        )
        all_old_embeds = mean_embeds + extra_embeds
        try:
            embedding_matrix = np.concatenate(
                [
                    x._embeddings
                    for x in tracklet_list
                    if x._embeddings is not None and len(x._embeddings) > 0
                ]
            )
        except ValueError:
            embedding_matrix = []

        # clear out any None values that may have made it in
        track_id = [x for x in track_id if x is not None]
        all_old_embeds = [x for x in all_old_embeds if x is not None]
        return cls(
            track_id,
            detections,
            all_old_embeds,
            skip_self_similarity=skip_self_similarity,
            embedding_matrix=embedding_matrix,
        )

    @staticmethod
    def compare_tracklets(
        tracklet_1: Tracklet, tracklet_2: Tracklet, other_anchors: bool = False
    ):
        """Compares embeddings between 2 tracklets.

        Args:
                tracklet_1: first tracklet to compare
                tracklet_2: second tracklet to compare
                other_anchors: whether or not to include additional anchors when tracklets are merged
        Returns:

        """
        embed_1 = [tracklet_1.mean_embed] if tracklet_1.mean_embed is not None else []
        embed_2 = [tracklet_2.mean_embed] if tracklet_2.mean_embed is not None else []

        if other_anchors:
            embed_1 = embed_1 + tracklet_1.additional_embeds
            embed_2 = embed_2 + tracklet_2.additional_embeds

        if len(embed_1) == 0 or len(embed_2) == 0:
            raise ValueError("Tracklets do not contain valid embeddings to compare.")

        return scipy.spatial.distance.cdist(embed_1, embed_2, metric="cosine")

    @property
    def frames(self):
        """Frames in which the tracklet is alive."""
        return self._frames

    @property
    def n_frames(self):
        """Number of frames the tracklet is alive."""
        return self._n_frames

    @property
    def start_frame(self):
        """The first frame the track exists."""
        return self._start_frame

    @property
    def end_frame(self):
        """The last frame the track exists."""
        return self._end_frame

    @property
    def track_id(self):
        """Track id assigned when constructed."""
        return self._track_id

    @property
    def mean_embed(self):
        """Mean embedding location of the tracklet."""
        return self._mean_embed

    @property
    def detection_list(self):
        """List of detections that are included in this tracklet."""
        return self._detection_list

    @property
    def additional_embeds(self):
        """List of additional embedding anchors that exist within this tracklet."""
        return self._additional_embeds

    @property
    def tracklet_self_similarity(self):
        """Self-similarity value for this tracklet."""
        return self._tracklet_self_similarity

    def overlaps_with(self, other: Tracklet) -> bool:
        """Returns if a tracklet overlaps with another.

        Args:
                other: the other tracklet.

        Returns:
                boolean whether these tracklets overlap
        """
        overlaps = np.intersect1d(self._frames, other.frames)
        return len(overlaps) > 0

    def compare_to(
        self, other: Tracklet, other_anchors: bool = True, default_distance: float = 0.5
    ) -> float:
        """Calculates the cost associated with matching this tracklet to another.

        Args:
                other: the other tracklet.
                other_anchors: bool to include other anchors in possible distances
                default_distance: cost returned if the tracklets can be linked, but either tracklet has no embedding to include

        Returns:
                cosine distance of this tracklet being the same mouse as another tracklet
        """
        # Check if the 2 tracklets overlap in time. If they do, don't provide a distance
        if self.overlaps_with(other):
            return None

        try:
            cosine_distance = self.compare_tracklets(self, other, other_anchors)
        # embeddings weren't comparible...
        except ValueError:
            return default_distance

        # Clip to safe -log probability values (if downstream requires)
        cosine_distance = np.clip(cosine_distance, 0, 1.0 - 1e-8)
        return np.min(cosine_distance)


class Fragment:
    """A collection of tracklets that overlap in time."""

    def __init__(
        self,
        tracklets: list[Tracklet],
        expected_distance: float = 0.15,
        length_target: int = 100,
        include_length_quality: bool = False,
    ):
        """Initializes a fragment object.

        Args:
                tracklets: List of tracklets belonging to the fragment
                expected_distance: Distance value observed when training identity to use
                length_target: Length of tracklets to priotize keeping
                include_length_quality: Instructs the quality to include length as a factor for quality
        """
        self._tracklets = tracklets
        self._tracklet_ids = list(
            chain.from_iterable([x.track_id for x in self._tracklets])
        )
        self._avg_frames = np.mean([x.n_frames for x in self._tracklets])
        self._tracklet_self_consistancies = np.asarray(
            [x.tracklet_self_similarity for x in self._tracklets]
        )
        self._tracklet_lengths = np.asarray([x.n_frames for x in self._tracklets])
        self._quality = self._generate_quality(
            expected_distance, length_target, include_length_quality
        )

    @classmethod
    def from_tracklets(
        cls,
        tracklets: list[Tracklet],
        global_count: int,
        expected_distance: float = 0.15,
        length_target: int = 100,
        include_length_quality: bool = False,
    ) -> list[Fragment]:
        """Generates a list of global fragments given tracklets that overlap.

        Args:
                tracklets: List of tracklets that can overlap in time
                global_count: count of tracklets that must exist at the same time to be considered global
                expected_distance: Distance value observed when training identity to use
                length_target: Length of tracklets to priotize keeping
                include_length_quality: Instructs the quality to include length as a factor for quality

        Returns:
                list of global fragments

        Notes:
                We use an undirected graph to generate global fragments. We can generate an undirected graph where each tracklet is a node and whether a node overlaps with another is an edge. Cliques with global_count number of nodes are a valid global fragment.
        """
        edges = []
        for i, tracklet_1 in enumerate(tracklets):
            for j, tracklet_2 in enumerate(tracklets):
                if i <= j:
                    continue
                # skip 1-frame tracklets
                # if tracklet_1.n_frames <= 1 or tracklet_2.n_frames <= 1:
                # 	continue
                if tracklet_1.overlaps_with(tracklet_2):
                    edges.append((i, j))

        graph = nx.Graph()
        graph.add_edges_from(edges)

        global_fragments = []
        for cur_clique in nx.enumerate_all_cliques(graph):
            if len(cur_clique) < global_count:
                continue
            # since enumerate_all_cliques yields cliques sorted by size
            # the first one that is larger means we're done
            if len(cur_clique) > global_count:
                break
            global_fragments.append(
                Fragment(
                    [tracklets[i] for i in cur_clique],
                    expected_distance,
                    length_target,
                    include_length_quality,
                )
            )

        return global_fragments

    @property
    def quality(self):
        """Quality of the global fragment. See `_generate_quality`."""
        return self._quality

    @property
    def tracklet_ids(self):
        """List of all tracklet ids contained in this fragment. If a tracklet was merged, all ids are included, so this list may be longer than the number of tracklets."""
        return self._tracklet_ids

    @property
    def avg_frames(self):
        """Average frames each tracklet exists in this fragment."""
        return self._avg_frames

    def _generate_quality(
        self, expected_distance, length_target, include_length: bool = False
    ):
        """Calculates the quality metric of this global fragment.

        Args:
                expected_distance: Distance value observed when training identity
                length_target: Length of tracklets to prioritize keeping
                include_length: Instructs the quality to include length as a factor

        Returns:
                Quality of this fragment. Value scales between 0-1 with 1 indicating high quality and 0 indicating lowest quality.

        Fragment quality is based on 2 or 3 factors multiplied, depending upon include_length value:
                1. Percent of tracklets that pass the self-consistancy vs length test. The self-consistancy test is the mean cosine distance relative to the mean within the tracklet / expected distance is < length of tracklet / important tracklet length.
                2. Mean distance between the tracklets
                (3.) Average length of the tracklets
        Terms 1 and 2 scale between 0-1. Term 3 is unbounded.
        """
        percent_good_tracklets = np.mean(
            self._tracklet_self_consistancies / expected_distance
            < self._tracklet_lengths / length_target
        )
        try:
            tracklet_distances = []
            for i in range(len(self._tracklets)):
                for j in range(len(self._tracklets)):
                    if i < j:
                        tracklet_distances.append(
                            Tracklet.compare_tracklets(
                                self._tracklets[i], self._tracklets[j]
                            )
                        )
        # ValueError is raised if one of the tracklets doesn't have embeddings (e.g. no frames in it had an embedding value)
        except ValueError:
            return 0.0

        quality_value = percent_good_tracklets * np.clip(
            np.mean(tracklet_distances), 0, 1
        )
        if include_length:
            quality_value *= self._avg_frames
        return quality_value

    def overlaps_with(self, other: Fragment):
        """Identifies the number of overlapping tracklets between 2 fragments.

        Args:
                other: The other fragment to compare to

        Returns:
                count of tracklets common between the two fragments
        """
        overlaps = 0
        for t1 in self._tracklets:
            for t2 in other._tracklets:
                if np.any(np.asarray(t1.track_id) == np.asarray(t2.track_id)):
                    overlaps += 1
        return overlaps

    def hungarian_match(self, other: Fragment, other_anchors: bool = False):
        """Applies hungarian matching of tracklets between this fragment and another.

        Args:
                other: The other fragment to compare to
                other_anchors: If one of the tracklets was merged, do we allow original anchors to be used for cost?

        Returns:
                tuple of (matches, total_cost)
                matches: List of tuples of tracklets that were matched.
                total_cost: Total cost associated with the matching
        """
        tracklet_distances = np.zeros([len(self._tracklets), len(other._tracklets)])
        for i, t1 in enumerate(self._tracklets):
            for j, t2 in enumerate(other._tracklets):
                if Tracklet.overlaps_with(t1, t2) and not np.any(
                    np.asarray(t1.track_id) == np.asarray(t2.track_id)
                ):
                    # Note: we can't use np.inf here because linear_sum_assignment fails, so just use a large value
                    # `Tracklet.compare_tracklets` should be bound by 0-1, so 1000 should be large enough
                    tracklet_distances[i, j] = 1000
                else:
                    try:
                        tracklet_distances[i, j] = Tracklet.compare_tracklets(
                            t1, t2, other_anchors=other_anchors
                        )
                    # If tracklets don't have embeddings to compare, give it a cost lower than overlapping, but still large
                    except ValueError:
                        tracklet_distances[i, j] = 100
        self_idxs, other_idxs = scipy.optimize.linear_sum_assignment(tracklet_distances)

        matches = [
            (self._tracklets[i], other._tracklets[j])
            for i, j in zip(self_idxs, other_idxs, strict=False)
        ]
        total_cost = np.sum(
            [
                tracklet_distances[i, j]
                for i, j in zip(self_idxs, other_idxs, strict=False)
            ]
        )

        return matches, total_cost


class VideoObservations:
    """Object that manages observations within a video to match them."""

    def __init__(self, observations: list[list[Detection]]):
        """Initializes a VideoObservation object.

        Args:
                observations: list of list of detections. See `read_pose_detections` static method.
        """
        # Observation and tracklet data that stores primary information about what is being linked.
        self._observations = observations
        self._tracklets = None

        # Dictionaries that store how observations and tracks get assigned an ID
        # Dict of dicts where self._observation_id_dict[frame_key][observation_key] stores tracklet_id
        self._observation_id_dict = None
        # Dict where self._stitch_translation[tracklet_id] stores longterm_id
        self._stitch_translation = None

        # Metadata
        self._num_frames = len(observations)
        self._median_observation = int(np.median([len(x) for x in observations]))
        # Add 0.5 to do proper rounding with int cast
        self._avg_observation = int(np.mean([len(x) for x in observations]) + 0.5)
        self._tracklet_gen_method = None
        self._tracklet_stitch_method = None

        self._pool = None

    @property
    def num_frames(self):
        """Number of frames."""
        return self._num_frames

    @property
    def tracklet_gen_method(self):
        """Method used in generating tracklets."""
        return self._tracklet_gen_method

    @property
    def tracklet_stitch_method(self):
        """Method used in stitching tracklets."""
        return self._tracklet_stitch_method

    @property
    def stitch_translation(self):
        """Translation dictionary, only available after stitching."""
        if self._stitch_translation is None:
            warnings.warn(
                "No stitching has been applied. Returning empty translation.",
                stacklevel=2,
            )
            return {}
        return self._stitch_translation.copy()

    @classmethod
    def from_pose_file(cls, pose_file, match_tolerance: float = 0):
        """Initializes a VideoObservation object from a pose file using `read_pose_detections`."""
        return cls(cls.read_pose_detections(pose_file, match_tolerance))

    @staticmethod
    def read_pose_detections(pose_file, match_tolerance: float = 0) -> list:
        """Reads and matches poses with segmentation from a pose file.

        Args:
                pose_file: filename for the pose
                match_tolerance: tolerance for matching segmentation with pose. 0 indicates average inside segmentation with negative indicating allowing more outside.

        Returns:
                list of lists of Detections where the first level of list is frames and the second level is observations within a frame
        """
        observations = []
        with h5py.File(pose_file, "r") as f:
            all_poses = f["poseest/points"][:]
            all_embeds = f["poseest/identity_embeds"][:]
            all_segs = segs = f["poseest/seg_data"][:]
        for frame in np.arange(all_poses.shape[0]):
            poses = all_poses[frame]
            embeds = all_embeds[frame]
            valid_poses = ~np.all(np.all(poses == 0, axis=-1), axis=-1)
            pose_idxs = np.where(valid_poses)[0]
            embeds = embeds[valid_poses]
            poses = poses[valid_poses]
            segs = all_segs[frame]
            valid_segs = ~np.all(np.all(np.all(segs == -1, axis=-1), axis=-1), axis=-1)
            seg_idxs = np.where(valid_segs)[0]
            segs = segs[valid_segs]
            matches = hungarian_match_points_seg(poses, segs, max_dist=match_tolerance)
            frame_observations = []
            for cur_pose in np.arange(len(poses)):
                if cur_pose in matches[:, 0]:
                    matched_seg = matches[:, 1][matches[:, 0] == cur_pose][0]
                    frame_observations.append(
                        Detection(
                            frame,
                            pose_idxs[cur_pose],
                            poses[cur_pose],
                            embeds[cur_pose],
                            seg_idxs[matched_seg],
                            segs[matched_seg],
                        )
                    )
                else:
                    frame_observations.append(
                        Detection(
                            frame,
                            pose_idxs[cur_pose],
                            poses[cur_pose],
                            embeds[cur_pose],
                        )
                    )
            observations.append(frame_observations)
        return observations

    def get_id_mat(
        self, pose_shape: list[int] | None = None, seg_shape: list[int] | None = None
    ) -> np.ndarray:
        """Generates identity matrices to store in a pose file.

        Args:
                pose_shape: shape of pose id data of shape [frames, max_poses]
                seg_shape: shape of seg id data [frames, max_segs]

        Returns:
                tuple of (pose_mat, seg_mat)
                pose_mat: matrix of pose identities
                seg_mat: matrix of segmentation identities
        """
        if self._observation_id_dict is None:
            raise ValueError(
                "Tracklets not generated yet, cannot return tracklet matrix."
            )

        if pose_shape is None:
            n_frames = len(self._observations)
            # TODO:
            # This currently fails when there is a frame with 0 observations (eg start/end of experiment).
            # Send pose_shape and seg_shape in these cases
            max_poses = np.nanmax(
                [
                    np.nanmax(
                        [
                            x.pose_idx if x.pose_idx is not None else np.nan
                            for x in frame_observations
                        ]
                    )
                    for frame_observations in self._observations
                ]
            )
            pose_shape = [n_frames, int(max_poses + 1)]
        assert len(pose_shape) == 2
        pose_id_mat = np.zeros(pose_shape, dtype=np.int32)

        if seg_shape is None:
            n_frames = len(self._observations)
            max_segs = np.nanmax(
                [
                    np.nanmax(
                        [
                            x.seg_idx if x.seg_idx is not None else np.nan
                            for x in frame_observations
                        ]
                    )
                    for frame_observations in self._observations
                ]
            )
            seg_shape = [n_frames, int(max_segs + 1)]
        assert len(seg_shape) == 2
        seg_id_mat = np.zeros(seg_shape, dtype=np.int32)
        #
        max_track_id = np.max(
            [
                np.max(list(x.values())) if len(x) > 0 else 0
                for x in self._observation_id_dict.values()
            ]
        )

        cur_unassigned_track_id = max_track_id + 1
        for cur_frame in np.arange(len(self._observations)):
            for obs_index, cur_observation in enumerate(self._observations[cur_frame]):
                assigned_id = self._observation_id_dict.get(cur_frame, {}).get(
                    obs_index, cur_unassigned_track_id
                )
                if assigned_id == cur_unassigned_track_id:
                    cur_unassigned_track_id += 1
                if cur_observation.pose_idx is not None:
                    pose_id_mat[cur_frame, cur_observation.pose_idx] = assigned_id + 1
                if cur_observation.seg_idx is not None:
                    seg_id_mat[cur_frame, cur_observation.seg_idx] = assigned_id + 1
        return pose_id_mat, seg_id_mat

    def get_embed_centers(self):
        """Calculates the embedding centers for each longterm ID.

        Returns:
                center embedding data of shape [n_ids, embed_dim]
        """
        if self._tracklets is None or self._stitch_translation is None:
            raise ValueError(
                "Tracklet stitching not yet conducted. Cannot calculate centers."
            )

        embedding_shape = self._tracklets[0].mean_embed.shape
        longterm_ids = np.asarray(list(set(self._stitch_translation.values())))
        longterm_ids = longterm_ids[longterm_ids != 0]

        # To calculate an average for merged tracklets, we weight by number of frames
        longterm_data = {}
        for cur_tracklet in self._tracklets:
            # Dangerous, but these tracklets are supposed to only have 1 track_id value
            track_id = cur_tracklet.track_id[0]
            if track_id not in list(self._stitch_translation.keys()):
                continue
            longterm_id = self._stitch_translation[track_id]
            n_frames = cur_tracklet.n_frames
            embed_value = cur_tracklet.mean_embed
            id_frame_counts, id_embeds = longterm_data.get(longterm_id, ([], []))
            id_frame_counts.append(n_frames)
            id_embeds.append(embed_value)
            longterm_data[longterm_id] = (id_frame_counts, id_embeds)

        # Calculate the weighted average
        embedding_centers = np.zeros([np.max(longterm_ids), embedding_shape[0]])
        for longterm_id, (frame_counts, embeddings) in longterm_data.items():
            mean_embed = np.average(np.stack(embeddings), axis=0, weights=frame_counts)
            embedding_centers[int(longterm_id - 1)] = mean_embed

        return embedding_centers

    def _make_tracklets(self, include_unassigned: bool = True):
        """Updates internal tracklets in this object based on generated tracklets.

        Args:
                include_unassigned: if true, observations that are unassigned are added to tracklets of length 1.
        """
        if self._observation_id_dict is None:
            warnings.warn("Tracklets not generated.", stacklevel=2)
            return
        # observation dictionary is frames -> observation_num -> id
        # tracklets need to be id -> list of observations
        tracklet_dict = {}
        unmatched_observations = []
        for frame, frame_observations in self._observation_id_dict.items():
            for observation_num, observation_id in frame_observations.items():
                observation_list = tracklet_dict.get(observation_id, [])
                observation_list.append(self._observations[frame][observation_num])
                tracklet_dict[observation_id] = observation_list
            available_observations = range(len(self._observations[frame]))
            unassigned_observations = [
                x for x in available_observations if x not in frame_observations
            ]
            for observation_num in unassigned_observations:
                unmatched_observations.append(
                    self._observations[frame][observation_num]
                )

        # Construct the tracklets
        tracklet_list = []
        for tracklet_id, observation_list in tracklet_dict.items():
            tracklet_list.append(Tracklet(tracklet_id, observation_list))

        if include_unassigned:
            cur_tracklet_id = np.max(np.asarray(list(tracklet_dict.keys())))
            for cur_observation in unmatched_observations:
                tracklet_list.append(Tracklet(int(cur_tracklet_id), [cur_observation]))
                cur_tracklet_id += 1

        self._tracklets = tracklet_list

    def _get_transition_costs(
        self,
        all_comparisons: bool = True,
        include_inf: bool = True,
        longer_track_priority: float = 0.0,
        longer_track_length: float = 100,
    ) -> dict:
        """Calculate cost associated with linking any pair of tracks.

        Args:
                all_comparisons: include comparisons of original embed centers before merges (if tracklets include merges)
                include_inf: return a completed dictionary with np.inf placed in locations where tracklets cannot be merged
                longer_track_priority: multiplier for prioritizing longer tracklets over shorter ones. 0 indicates no adjustment and positive values indicate more priority for longer tracklets. At a value of 1, tracklets longer than longer_track_length will be merged before those shorter
                longer_track_length: value at which longer tracks get prioritized

        Note:
                Transitions are a dictionary of link costs where transitions[id1][id2] = cost
                IDs are sorted to reduce memory footprint such that id1 < id2
        """
        transitions = {}
        for i, current_track in enumerate(self._tracklets):
            for j, other_track in enumerate(self._tracklets):
                # Only do 1 pairwise comparison, enforce i is always less than j
                if i >= j:
                    continue
                match_cost = current_track.compare_to(
                    other_track, other_anchors=all_comparisons
                )
                # adjustment for track lengths
                if match_cost is not None and longer_track_priority != 0:
                    sigmoid_length_current = 1 / (
                        1 + np.exp(longer_track_length - current_track.n_frames)
                    )
                    sigmoid_length_other = 1 / (
                        1 + np.exp(longer_track_length - other_track.n_frames)
                    )
                    match_cost += (
                        1 - sigmoid_length_current * sigmoid_length_other
                    ) * longer_track_priority
                match_costs = transitions.get(i, {})
                if match_cost is not None and not np.isinf(match_cost):
                    match_costs[j] = match_cost
                else:
                    if include_inf:
                        match_costs[j] = np.inf
                transitions[i] = match_costs
        return transitions

    def _start_pool(self, n_threads: int = 1):
        """Starts the multiprocessing pool.

        Args:
                n_threads: number of threads to parallelize.
        """
        if self._pool is None:
            self._pool = multiprocessing.Pool(processes=n_threads)

    def _kill_pool(self):
        """Stops the multiprocessing pool."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def _calculate_costs(self, frame_1: int, frame_2: int, rotate_pose: bool = False):
        """Calculates the cost matrix between all observations in 2 frames using multiple threads.

        Args:
                frame_1: frame index 1 to compare
                frame_2: frame index 2 to compare
                rotate_pose: allow pose to be rotated 180 deg

        Returns:
                cost matrix
        """
        # Only use parallelism if the pool has been started.
        if self._pool is not None:
            out_shape = [
                len(self._observations[frame_1]),
                len(self._observations[frame_2]),
            ]
            xs, ys = np.meshgrid(range(out_shape[0]), range(out_shape[1]))

            xs = xs.flatten()
            ys = ys.flatten()
            chunks = [
                (
                    self._observations[frame_1][x],
                    self._observations[frame_2][y],
                    40,
                    0.0,
                    (1.0, 1.0, 1.0),
                    rotate_pose,
                )
                for x, y in zip(xs, ys, strict=False)
            ]

            results = self._pool.map(Detection.calculate_match_cost_multi, chunks)

            results = np.asarray(results).reshape(out_shape)
            return results

        # Non-parallel version
        match_costs = np.zeros(
            [len(self._observations[frame_1]), len(self._observations[frame_2])]
        )
        for i, cur_obs in enumerate(self._observations[frame_1]):
            cur_obs.cache()
            for j, next_obs in enumerate(self._observations[frame_2]):
                next_obs.cache()
                match_costs[i, j] = Detection.calculate_match_cost(
                    cur_obs, next_obs, pose_rotation=rotate_pose
                )
        return match_costs

    def _calculate_costs_vectorized(
        self, frame_1: int, frame_2: int, rotate_pose: bool = False
    ):
        """Vectorized version of cost calculation between observations in 2 frames.

        Args:
                frame_1: frame index 1 to compare
                frame_2: frame index 2 to compare
                rotate_pose: allow pose to be rotated 180 deg

        Returns:
                cost matrix computed using vectorized operations
        """
        # Extract features for both frames
        features1 = VectorizedDetectionFeatures(self._observations[frame_1])
        features2 = VectorizedDetectionFeatures(self._observations[frame_2])

        # Compute vectorized match costs using the same parameters as original
        return compute_vectorized_match_costs(
            features1,
            features2,
            max_dist=40,
            default_cost=0.0,
            beta=(1.0, 1.0, 1.0),
            pose_rotation=rotate_pose,
        )

    def generate_greedy_tracklets_vectorized(
        self, max_cost: float = -np.log(1e-3), rotate_pose: bool = False
    ):
        """Vectorized version of greedy tracklet generation for improved performance.

        Args:
                max_cost: negative log probability associated with the maximum cost that will be greedily matched.
                rotate_pose: allow pose to be rotated 180 deg when calculating distance cost
        """
        # Seed first values
        frame_dict = {0: {i: i for i in np.arange(len(self._observations[0]))}}
        cur_tracklet_id = len(self._observations[0])
        prev_matches = frame_dict[0]

        # Main loop to cycle over greedy matching.
        # Each match problem is posed as a bipartite graph between sequential frames
        for frame in np.arange(len(self._observations) - 1) + 1:
            # Calculate cost using vectorized method
            match_costs = self._calculate_costs_vectorized(
                frame - 1, frame, rotate_pose
            )

            # Use optimized greedy matching - O(k log k) instead of O(n³)
            matches = vectorized_greedy_matching(match_costs, max_cost)

            # Map the matches to tracklet IDs from previous frame
            tracklet_matches = {}
            for col_idx, row_idx in matches.items():
                tracklet_matches[col_idx] = prev_matches[row_idx]

            # Fill any unmatched observations with new tracklet IDs
            for j in range(len(self._observations[frame])):
                if j not in tracklet_matches:
                    tracklet_matches[j] = cur_tracklet_id
                    cur_tracklet_id += 1

            frame_dict[frame] = tracklet_matches
            prev_matches = tracklet_matches

        # Final modification of internal state
        self._observation_id_dict = frame_dict
        self._tracklet_gen_method = "greedy_vectorized"
        self._make_tracklets()

    def generate_greedy_tracklets_batched(
        self,
        max_cost: float = -np.log(1e-3),
        rotate_pose: bool = False,
        batch_size: int = 32,
    ):
        """Memory-efficient batched version of greedy tracklet generation.

        Uses BatchedFrameProcessor to handle large videos with controlled memory usage.

        Args:
                max_cost: negative log probability associated with the maximum cost that will be greedily matched.
                rotate_pose: allow pose to be rotated 180 deg when calculating distance cost
                batch_size: number of frames to process together in each batch
        """
        processor = BatchedFrameProcessor(batch_size=batch_size)
        frame_dict = processor.process_video_observations(self, max_cost, rotate_pose)

        # Final modification of internal state
        self._observation_id_dict = frame_dict
        self._tracklet_gen_method = "greedy_vectorized_batched"
        self._make_tracklets()

    def generate_greedy_tracklets(
        self,
        max_cost: float = -np.log(1e-3),
        rotate_pose: bool = False,
        num_threads: int = 1,
    ):
        """Applies a greedy technique of identity matching to a list of frame observations.

        Args:
                max_cost: negative log probability associated with the maximum cost that will be greedily matched.
                rotate_pose: allow pose to be rotated 180 deg when calculating distance cost
                num_threads: maximum number of threads to parallelize cost matrix calculation
        """
        # Seed first values
        frame_dict = {0: {i: i for i in np.arange(len(self._observations[0]))}}
        cur_tracklet_id = len(self._observations[0])
        prev_matches = frame_dict[0]

        if num_threads > 1:
            self._start_pool(num_threads)

        # Main loop to cycle over greedy matching.
        # Each match problem is posed as a bipartite graph between sequential frames
        for frame in np.arange(len(self._observations) - 1) + 1:
            # Cache the segmentation and rotation data
            for obs in self._observations[frame - 1]:
                obs.cache()
            for obs in self._observations[frame]:
                obs.cache()
            # Calculate cost and greedily match
            match_costs = self._calculate_costs(frame - 1, frame, rotate_pose)
            match_costs = np.ma.array(match_costs, fill_value=max_cost, mask=False)
            matches = {}
            while np.any(~match_costs.mask) and np.any(match_costs.filled() < max_cost):
                next_best = np.unravel_index(np.argmin(match_costs), match_costs.shape)
                matches[next_best[1]] = prev_matches[next_best[0]]
                match_costs.mask[next_best[0], :] = True
                match_costs.mask[:, next_best[1]] = True
            # Fill any unmatched observations
            for j in range(len(self._observations[frame])):
                if j not in matches:
                    matches[j] = cur_tracklet_id
                    cur_tracklet_id += 1
            frame_dict[frame] = matches
            # Cleanup for next loop iteration
            for cur_obs in self._observations[frame - 1]:
                cur_obs.clear_cache()
            prev_matches = matches
        if self._pool is not None:
            self._kill_pool()
        # Final modification of internal state
        self._observation_id_dict = frame_dict
        self._tracklet_gen_method = "greedy"
        self._make_tracklets()

    def stitch_greedy_tracklets_optimized(
        self,
        num_tracks: int | None = None,
        all_embeds: bool = True,
        prioritize_long: bool = False,
    ):
        """Optimized greedy method that links merges tracklets 1 at a time based on lowest cost.

        Args:
                        num_tracks: number of tracks to produce
                        all_embeds: bool to include original tracklet centers as merges are made
                        prioritize_long: bool to adjust cost of linking with length of tracklets

        Notes:
                        Optimized version eliminates O(n³) pandas DataFrame recreation bottleneck.
                        Uses numpy arrays and incremental cost matrix updates for O(n²) complexity.
        """
        if num_tracks is None:
            num_tracks = self._avg_observation

        # copy original tracklet list, so that we can revert at the end
        original_tracklets = self._tracklets

        # Early exit if no tracklets or only one tracklet
        if len(self._tracklets) <= 1:
            self._stitch_translation = {0: 0}
            self._tracklets = original_tracklets
            self._tracklet_stitch_method = "greedy"
            return

        # Get initial transition costs as dict and convert to numpy matrix
        cost_dict = self._get_transition_costs(
            all_embeds, True, longer_track_priority=float(prioritize_long)
        )

        # Build numpy cost matrix - work with a copy of tracklets for merging
        working_tracklets = list(
            self._tracklets
        )  # Copy for modifications during merging
        n_tracklets = len(working_tracklets)

        # Initialize cost matrix with infinity
        cost_matrix = np.full((n_tracklets, n_tracklets), np.inf, dtype=np.float64)

        # Fill cost matrix from cost_dict
        for i, costs_for_i in cost_dict.items():
            for j, cost in costs_for_i.items():
                cost_matrix[i, j] = cost
                cost_matrix[j, i] = cost  # Matrix should be symmetric

        # Track which tracklets are still active (not merged)
        active_tracklets = set(range(n_tracklets))

        # Main stitching loop - continues until no more valid merges
        while len(active_tracklets) > 1:
            # Find minimum cost among active tracklets
            min_cost = np.inf
            best_pair = None

            for i in active_tracklets:
                for j in active_tracklets:
                    if i < j and cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        best_pair = (i, j)

            # If no finite cost found, break (no more valid merges)
            if best_pair is None or np.isinf(min_cost):
                break

            tracklet_1_idx, tracklet_2_idx = best_pair

            # Create new merged tracklet
            new_tracklet = Tracklet.from_tracklets(
                [working_tracklets[tracklet_1_idx], working_tracklets[tracklet_2_idx]],
                True,
            )

            # Remove merged tracklets from active set
            active_tracklets.remove(tracklet_1_idx)
            active_tracklets.remove(tracklet_2_idx)

            # Add new tracklet to working list and get its index
            working_tracklets.append(new_tracklet)
            new_tracklet_idx = len(working_tracklets) - 1
            active_tracklets.add(new_tracklet_idx)

            # Extend cost matrix for new tracklet if needed
            if new_tracklet_idx >= cost_matrix.shape[0]:
                # Extend matrix size
                old_size = cost_matrix.shape[0]
                new_size = max(old_size * 2, new_tracklet_idx + 1)
                new_matrix = np.full((new_size, new_size), np.inf, dtype=np.float64)
                new_matrix[:old_size, :old_size] = cost_matrix
                cost_matrix = new_matrix

            # Calculate costs for new tracklet with all remaining active tracklets
            for other_idx in active_tracklets:
                if other_idx != new_tracklet_idx and other_idx < len(working_tracklets):
                    # Calculate cost between new tracklet and existing tracklet
                    match_cost = new_tracklet.compare_to(
                        working_tracklets[other_idx], other_anchors=all_embeds
                    )

                    # Apply priority adjustment if enabled
                    if match_cost is not None and prioritize_long:
                        longer_track_length = 100  # Default from _get_transition_costs
                        sigmoid_length_new = 1 / (
                            1 + np.exp(longer_track_length - new_tracklet.n_frames)
                        )
                        sigmoid_length_other = 1 / (
                            1
                            + np.exp(
                                longer_track_length
                                - working_tracklets[other_idx].n_frames
                            )
                        )
                        match_cost += (
                            1 - sigmoid_length_new * sigmoid_length_other
                        ) * float(prioritize_long)

                    # Update cost matrix
                    if match_cost is not None and not np.isinf(match_cost):
                        cost_matrix[new_tracklet_idx, other_idx] = match_cost
                        cost_matrix[other_idx, new_tracklet_idx] = match_cost
                    else:
                        cost_matrix[new_tracklet_idx, other_idx] = np.inf
                        cost_matrix[other_idx, new_tracklet_idx] = np.inf

        # Update self._tracklets with the merged result for ID assignment
        self._tracklets = [working_tracklets[i] for i in active_tracklets]

        # Tracklets are formed. Now we should assign the longest ones IDs.
        tracklet_lengths = [len(x.frames) for x in self._tracklets]
        assignment_order = np.argsort(tracklet_lengths)[::-1]
        track_to_longterm_id = {0: 0}
        current_id = num_tracks
        for cur_assignment in assignment_order:
            ids_to_assign = self._tracklets[cur_assignment].track_id
            for cur_tracklet_id in ids_to_assign:
                track_to_longterm_id[int(cur_tracklet_id + 1)] = (
                    current_id if current_id > 0 else 0
                )
            current_id -= 1

        self._stitch_translation = track_to_longterm_id
        self._tracklets = original_tracklets
        self._tracklet_stitch_method = "greedy"

    def stitch_greedy_tracklets(
        self,
        num_tracks: int | None = None,
        all_embeds: bool = True,
        prioritize_long: bool = False,
    ):
        """Greedy method that links merges tracklets 1 at a time based on lowest cost.

        Args:
                num_tracks: number of tracks to produce
                all_embeds: bool to include original tracklet centers as merges are made
                prioritize_long: bool to adjust cost of linking with length of tracklets
        """
        if num_tracks is None:
            num_tracks = self._avg_observation

        # copy original tracklet list, so that we can revert at the end
        original_tracklets = self._tracklets

        # We can use pandas to do slightly easier searching
        current_costs = pd.DataFrame(
            self._get_transition_costs(
                all_embeds, True, longer_track_priority=float(prioritize_long)
            )
        )
        while not np.all(np.isinf(current_costs.to_numpy(na_value=np.inf))):
            t1, t2 = np.unravel_index(
                np.argmin(current_costs.to_numpy(na_value=np.inf)), current_costs.shape
            )
            tracklet_1 = current_costs.index[t1]
            tracklet_2 = current_costs.columns[t2]
            new_tracklet = Tracklet.from_tracklets(
                [self._tracklets[tracklet_1], self._tracklets[tracklet_2]], True
            )
            self._tracklets = [
                x
                for i, x in enumerate(self._tracklets)
                if i not in [tracklet_1, tracklet_2]
            ] + [new_tracklet]
            current_costs = pd.DataFrame(
                self._get_transition_costs(
                    all_embeds, True, longer_track_priority=float(prioritize_long)
                )
            )

        # Tracklets are formed. Now we should assign the longest ones IDs.
        tracklet_lengths = [len(x.frames) for x in self._tracklets]
        assignment_order = np.argsort(tracklet_lengths)[::-1]
        track_to_longterm_id = {0: 0}
        current_id = num_tracks
        for cur_assignment in assignment_order:
            ids_to_assign = self._tracklets[cur_assignment].track_id
            for cur_tracklet_id in ids_to_assign:
                track_to_longterm_id[int(cur_tracklet_id + 1)] = (
                    current_id if current_id > 0 else 0
                )
            current_id -= 1

        self._stitch_translation = track_to_longterm_id
        self._tracklets = original_tracklets
        self._tracklet_stitch_method = "greedy"
