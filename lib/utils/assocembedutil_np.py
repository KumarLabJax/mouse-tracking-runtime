import math
import statistics
import numpy as np
from utils.assocembedutil import PoseInstance
import cv2


def localmax2D(data, min_thresh, min_dist):
    """
    Finds the local maxima for the given data arr.

    Parameters:
        data (numpy arr): the arr that we search for local maxima. Each 2D
        (rows, cols) slice will be searched for local maxima. If neighboring pixels
        have the same value we follow a simple tie breaking algorithm. The tie will
        be broken by considering the pixel with the larger row index greater, or if the
        rows are also equal, the pixel with the larger column index is considered
        greater.

        min_thresh (number): any pixels below this threshold will be excluded
        from the local maxima search

        min_dist (integer): minimum neighborhood size in pixels. We search a square
        neighborhood around each pixel min_dist pixels out. If a pixel is the largest
        value in this neighborhood we mark it a local maxima. All pixels within
        min_dist of the 2D boundary are excluded from the local maxima search. This
        parameter must be >= 1.

    Returns:
        A boolean arr with the same shape as data and on the same device. Elements
        will be True where a local maxima was detected and False elsewhere.
    """

    assert min_dist >= 1
    assert np.ndim(data) == 3

    result_stack = []
    for i in np.arange(np.shape(data)[0]):
        point_heatmap = np.copy(data[i,:,:])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_dist*2+1, min_dist*2+1))
        # Peakfinding
        dilated = cv2.dilate(point_heatmap, kernel)
        mask = point_heatmap >= dilated
        # Non-max suppression
        eroded = cv2.erode(point_heatmap, kernel)
        mask_2 = point_heatmap > eroded
        mask = np.logical_and(mask, mask_2)
        # Threshold
        mask = np.logical_and(mask, point_heatmap>min_thresh)
        bool_arr = np.zeros_like(dilated, dtype=bool)+1
        bool_arr[~mask] = 0
        result_stack.append(bool_arr)
    return np.stack(result_stack)


def xy_dist(pt1, pt2):
    x_diff = pt2['x_pos'] - pt1['x_pos']
    y_diff = pt2['y_pos'] - pt1['y_pos']

    return math.sqrt(x_diff ** 2 + y_diff ** 2)

def calc_pose_instances(
        pose_heatmaps,
        pose_localmax,
        pose_embed_maps,
        min_embed_sep_between,
        max_embed_sep_within,
        max_inst_dist):

    """
    Given the input parameters for a single image/frame, return a list
    of PoseInstance objects

    Parameters:
        pose_heatmaps (numpy arr): contains 2D heatmaps representing confidence
        that a pose keypoint is detected at the respective 2D pixel locations.
        The shape of this tensor should be (joint_count, pixel_rows, pixel_columns)

        pose_localmax (numpy arr): this is a boolean arr with the same shape as
        pose_heatmaps. Each true value locates a local maxima in pose_heatmaps

        pose_embed_maps (numpy arr): the same shape as pose_heatmaps. This arr
        contains instance embedding values as described in "Associative Embedding:
        End-to-End Learning for Joint Detection and Grouping" (Newell et al.)

        min_embed_sep_between (number): minimum separation in the embedding space required
        between instances that are in close proximity

        max_embed_sep_within (number): maximum separation in the embedding space allowed
        within an instance

        max_inst_dist (number): the maximum distance is pixel units for neighboring
        keypoints of an instance

    Returns:
        A list of PoseInstace objects
    """

    joint_count = pose_heatmaps.shape[0]

    pose_instances = []
    for joint_index in range(joint_count):
        joint_localmax = pose_localmax[joint_index, ...]

        # Early skip to next joint if no joints predicted
        if np.count_nonzero(joint_localmax)==0:
            continue

        joint_xy = np.array(joint_localmax.nonzero())
        joint_xy = np.transpose(joint_xy, [1,0])

        joint_insts = []
        for inst_index in range(joint_xy.shape[0]):
            joint_insts.append({
                'joint_index': joint_index,
                'x_pos': joint_xy[inst_index, 1],
                'y_pos': joint_xy[inst_index, 0],
                'conf': pose_heatmaps[joint_index, joint_xy[inst_index, 0], joint_xy[inst_index, 1]],
                'embed': pose_embed_maps[joint_index, joint_xy[inst_index, 0], joint_xy[inst_index, 1]],
            })

        # Here we remove any keypoints that are both spatially too close and too
        # close in the embedding space. In these cases the joint with higher confidence
        # is kept and the other is discarded
        joint_insts.sort(key=lambda j: j['conf'])
        joint_insts_filtered = []
        for inst_index1, joint_inst1 in enumerate(joint_insts):
            min_embed_sep_violated = False
            for joint_inst2 in joint_insts[inst_index1 + 1:]:
                if (abs(joint_inst1['embed'] - joint_inst2['embed']) < min_embed_sep_between
                        and xy_dist(joint_inst1, joint_inst2) <= max_inst_dist):
                    min_embed_sep_violated = True
                    break
            if not min_embed_sep_violated:
                joint_insts_filtered.append(joint_inst1)
        joint_insts_filtered.reverse()
        joint_insts = joint_insts_filtered

        # we look at all valid combinations of joints with pose instances and
        # we prioritize by embedding distance
        candidate_keypoint_assignments = []
        for keypoint_index, curr_joint in enumerate(joint_insts):
            for pose_index, curr_pose_instance in enumerate(pose_instances):

                max_inst_dist_violated = True
                for pose_inst_pt in curr_pose_instance.keypoints.values():
                    if xy_dist(curr_joint, pose_inst_pt) <= max_inst_dist:
                        max_inst_dist_violated = False
                        break

                #embedding_dist = abs(curr_pose_instance.mean_inst_embed - curr_joint['embed'])
                embedding_dist = curr_pose_instance.weighted_embed_dist(curr_joint)
                if not max_inst_dist_violated and embedding_dist < max_embed_sep_within:
                    candidate_keypoint_assignments.append(
                        (pose_index, keypoint_index, embedding_dist))

        unassigned_keypoint_indexes = set(range(len(joint_insts)))
        candidate_keypoint_assignments.sort(key=lambda x: x[2])
        for pose_index, keypoint_index, embedding_dist in candidate_keypoint_assignments:
            curr_pose_instance = pose_instances[pose_index]
            if (keypoint_index in unassigned_keypoint_indexes
                    and joint_index not in curr_pose_instance.keypoints):
                curr_pose_instance.add_keypoint(joint_insts[keypoint_index])
                unassigned_keypoint_indexes.remove(keypoint_index)

        for keypoint_index in unassigned_keypoint_indexes:
            pose_instance = PoseInstance()
            pose_instance.add_keypoint(joint_insts[keypoint_index])
            pose_instances.append(pose_instance)

    return pose_instances


def pose_distance(pose1, pose2):

    """
    Calculate an averaged pixel distance between two poses that can be used for pose tracking.
    """

    # TODO if this isn't good enough we should correct distance using keypoint speed
    # to estimate position

    # total_distance = 0
    # point_count = 0

    # for joint_index, pose1_keypoint in pose1.keypoints.items():
    #     if joint_index in pose2.keypoints:
    #         pose2_keypoint = pose2.keypoints[joint_index]
    #         total_distance += xy_dist(pose1_keypoint, pose2_keypoint)
    #         point_count += 1

    # if point_count >= 1:
    #     return total_distance / point_count
    # else:
    #     return math.inf

    point_dists = []

    for joint_index, pose1_keypoint in pose1.keypoints.items():
        if joint_index in pose2.keypoints:
            pose2_keypoint = pose2.keypoints[joint_index]
            point_dists.append(xy_dist(pose1_keypoint, pose2_keypoint))

    if point_dists:
        return statistics.median(point_dists)
    else:
        return math.inf
