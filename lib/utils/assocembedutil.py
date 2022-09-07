import math
import statistics
import torch
import numpy as np

def localmax2D(data, min_thresh, min_dist):
    """
    Finds the local maxima for the given data tensor. All computationally intensive
    operations are handled by pytorch on the same device as the given data tensor.

    Parameters:
        data (tensor): the tensor that we search for local maxima. This tensor must
        have at least two dimensions (d1, d2, d3, ..., dn, rows, cols). Each 2D
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
        A boolean tensor with the same shape as data and on the same device. Elements
        will be True where a local maxima was detected and False elsewhere.
    """

    assert min_dist >= 1
    # Make sure the data is the correct shape
    # Allow 3 (single image) or 4 (batched images)
    orig_dim = data.dim()
    if data.dim() == 3:
        data = torch.unsqueeze(data, 0)
    assert data.dim() == 4

    # Peakfinding
    dilated = torch.nn.MaxPool2d(kernel_size=min_dist*2+1, stride=1, padding=min_dist)(data)
    mask = data >= dilated
    # Non-max suppression
    eroded = -torch.nn.MaxPool2d(kernel_size=min_dist*2+1, stride=1, padding=min_dist)(-data)
    mask_2 = data > eroded
    mask = torch.logical_and(mask, mask_2)
    # Threshold
    mask = torch.logical_and(mask, data>min_thresh)
    bool_arr = torch.zeros_like(dilated, dtype=bool)+1
    bool_arr[~mask] = 0
    if orig_dim == 3:
        bool_arr = torch.squeeze(bool_arr, 0)
    return bool_arr


def flatten_indices(indices_2d, shape):
    """
    This function will "flatten" the given index matrix such that it can be used
    as input to pytorch's `take` function.
    """

    # calculate the index multiplier vector that allows us to convert
    # from vector indicies to flat indices
    index_mult_vec = indices_2d.new_ones(len(shape))
    for i in range(len(shape) - 1):
        index_mult_vec[ : i + 1] *= shape[i + 1]

    return torch.sum(indices_2d * index_mult_vec, dim=1)


def xy_dist(pt1, pt2):
    x_diff = pt2['x_pos'] - pt1['x_pos']
    y_diff = pt2['y_pos'] - pt1['y_pos']

    return math.sqrt(x_diff ** 2 + y_diff ** 2)


class PoseInstance(object):

    def __init__(self):
        self.keypoints = dict()
        self.instance_track_id = 0
        self._sum_inst_embed = 0
        self._sum_inst_conf = 0

    @property
    def mean_inst_embed(self):
        return self._sum_inst_embed / len(self.keypoints)

    @property
    def mean_inst_conf(self):
        return self._sum_inst_conf / len(self.keypoints)

    def add_keypoint(self, keypoint):

        assert keypoint['joint_index'] not in self.keypoints
        self.keypoints[keypoint['joint_index']] = keypoint

        self._sum_inst_embed += keypoint['embed']
        self._sum_inst_conf += keypoint['conf']

    def nearest_dist(self, keypoint):
        min_dist = None
        for pose_keypoint in self.keypoints.values():
            curr_dist = xy_dist(keypoint, pose_keypoint)

            if min_dist is None or curr_dist < min_dist:
                min_dist = curr_dist

        return min_dist

    def weighted_embed_dist(self, keypoint):

        sum_of_weights = 0
        sum_of_weighted_embed_dists = 0

        for pose_keypoint in self.keypoints.values():
            curr_xy_dist = xy_dist(keypoint, pose_keypoint)
            curr_embed_dist = abs(keypoint['embed'] - pose_keypoint['embed'])
            if curr_xy_dist == 0:
                return curr_embed_dist

            sum_of_weighted_embed_dists += curr_embed_dist / curr_xy_dist
            sum_of_weights += 1.0 / curr_xy_dist

        assert sum_of_weights > 0

        return sum_of_weighted_embed_dists / sum_of_weights

    @staticmethod
    def from_xy_tensor(xy_tensor):
        pi = PoseInstance()
        pi.keypoints = {
            i: {'x_pos': x_pos, 'y_pos': y_pos}
            for i, (x_pos, y_pos) in enumerate(xy_tensor)
        }

        return pi


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
        pose_heatmaps (tensor): contains 2D heatmaps representing confidence
        that a pose keypoint is detected at the respective 2D pixel locations.
        The shape of this tensor should be (joint_count, pixel_rows, pixel_columns)

        pose_localmax (tensor): this is a boolean tensor with the same shape as
        pose_heatmaps. Each true value locates a local maxima in pose_heatmaps

        pose_embed_maps (tensor): the same shape as pose_heatmaps. This tensor
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

    joint_count = pose_heatmaps.size(0)

    pose_instances = []
    for joint_index in range(joint_count):
        joint_localmax = pose_localmax[joint_index, ...]

        # Early skip to next joint if no joints predicted
        if torch.count_nonzero(joint_localmax)==0:
            continue

        joint_xy = joint_localmax.nonzero()
        joint_xy = joint_xy.cpu().numpy().reshape([-1,2])

        joint_insts = []
        for inst_index in range(joint_xy.shape[0]):
            joint_insts.append({
                'joint_index': joint_index,
                'x_pos': joint_xy[inst_index, 1],
                'y_pos': joint_xy[inst_index, 0],
                'conf': pose_heatmaps[joint_index, joint_xy[inst_index, 0], joint_xy[inst_index, 1]].cpu().numpy(),
                'embed': pose_embed_maps[joint_index, joint_xy[inst_index, 0], joint_xy[inst_index, 1]].cpu().numpy(),
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

        # # TODO pick one of these two if/else blocks and delete the other
        # if False:
        #     for joint_inst in joint_insts:
        #         best_pose_match = None
        #         best_embed_diff = None

        #         # find nearest instance in embedding space
        #         for pose_instance in pose_instances:
        #             if joint_index not in pose_instance.keypoints:
        #                 embed_diff = abs(joint_inst['embed'] - pose_instance.mean_inst_embed)
        #                 if best_embed_diff is None or embed_diff < best_embed_diff:
        #                     spatial_dist = pose_instance.nearest_dist(joint_inst)
        #                     if spatial_dist <= max_inst_dist:
        #                         best_pose_match = pose_instance
        #                         best_embed_diff = embed_diff

        #         if best_pose_match is None:
        #             # since there's no existing pose match create a new one
        #             best_pose_match = PoseInstance()
        #             pose_instances.append(best_pose_match)

        #         best_pose_match.add_keypoint(joint_inst)
        # else:
        #     for pose_instance in pose_instances:
        #         best_keypoint_index = None
        #         best_embed_diff = None

        #         for keypoint_index, joint_inst in enumerate(joint_insts):
        #             embed_diff = abs(joint_inst['embed'] - pose_instance.mean_inst_embed)
        #             if best_embed_diff is None or embed_diff < best_embed_diff:
        #                 spatial_dist = pose_instance.nearest_dist(joint_inst)
        #                 if spatial_dist <= max_inst_dist:
        #                     best_keypoint_index = keypoint_index
        #                     best_embed_diff = embed_diff

        #         if best_keypoint_index is not None:
        #             best_keypoint = joint_insts[best_keypoint_index]
        #             del joint_insts[best_keypoint_index]
        #             pose_instance.add_keypoint(best_keypoint)

        #     for joint_inst in joint_insts:
        #         pose_instance = PoseInstance()
        #         pose_instance.add_keypoint(joint_inst)
        #         pose_instances.append(pose_instance)

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
