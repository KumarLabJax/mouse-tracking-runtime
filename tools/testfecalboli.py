import matplotlib
matplotlib.use("Agg")

import argparse
import colorsys
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import skimage.draw
import skimage.io
import torch
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import _init_paths
import utils.assocembedutil as aeutil
from config import cfg
from config import update_config

#from dataset.multimousepose import MultiPoseDataset, parse_poses, decompose_frame_name
from dataset.fecalbolidata import parse_fecal_boli_labels, FecalBoliDataset

import models

# Example use:
#
# python -u tools/testfecalboli.py \
#       --cvat-files data/fecal-boli/*.xml \
#       --image-dir data/fecal-boli/images \
#       --image-list data/fecal-boli/fecal-boli-val-set.txt \
#       --plot-heatmap \
#       --cfg experiments/fecalboli/fecalboli_2020-05-0-08.yaml \
#       --model-file output-fecal-boli/fecalboli/pose_hrnet/fecalboli_2020-05-0-08/best_state.pth \
#       --image-out-dir temp4
#
# for (( i=1; i<9; i++ )); do echo "i = $i"; python -u tools/testfecalboli.py \
#       --cvat-files data/fecal-boli/*.xml \
#       --image-dir data/fecal-boli/images \
#       --image-list data/fecal-boli/fecal-boli-val-set.txt \
#       --plot-heatmap \
#       --cfg "experiments/fecalboli/fecalboli_2020-06-19_0${i}.yaml" \
#       --model-file "output-fecal-boli/fecalboli/pose_hrnet/fecalboli_2020-06-19_0${i}/best_state.pth" \
#       --image-out-dir "temp${i}"; done

def main():

    parser = argparse.ArgumentParser(description='test the fecal boli detection network')

    parser.add_argument(
        '--cvat-files',
        help='list of CVAT XML files to use',
        nargs='+',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--image-dir',
        help='directory containing images',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--image-list',
        help='file containing newline separated list of images to use',
        default=None,
    )
    parser.add_argument(
        '--model-file',
        help='the model file to use for inference',
        required=True,
    )
    # parser.add_argument(
    #     '--confidence-threshold',
    #     help='minimum confidence threshold to test',
    #     default=0.0,
    #     type=float,
    # )
    parser.add_argument(
        '--cfg',
        help='the configuration for the model to use for inference',
        required=True,
        type=str,
    )
    # # TODO we should change this to cm units rather than pixels
    # parser.add_argument(
    #     '--max-inst-dist-px',
    #     help='maximum keypoint separation distance in pixels. For a keypoint to '
    #          'be added to an instance there must be at least one point in the '
    #          'instance which is within this number of pixels.',
    #     type=int,
    #     default=150,
    # )
    # parser.add_argument(
    #     '--max-embed-sep-within-instances',
    #     help='maximum embedding separation allowed for a joint to be added to an existing '
    #          'instance within the max distance separation',
    #     type=float,
    #     default=0.2,
    # )
    # parser.add_argument(
    #     '--min-embed-sep-between-instances',
    #     help='if two joints of the the same type (eg. both right ear) are within the max '
    #          'distance separation and their embedding separation doesn\'t meet or '
    #          'exceed this threshold only the point with the highest heatmap value is kept.',
    #     type=float,
    #     default=0.1,
    # )
    # parser.add_argument(
    #     '--min-pose-heatmap-val',
    #     type=float,
    #     default=0.4,
    # )
    parser.add_argument(
        '--image-out-dir',
        help='the directory we plot to',
    )
    parser.add_argument(
        '--plot-heatmap',
        action='store_true',
        help='indicates that the heatmap should be included in generated image output',
    )
    # parser.add_argument(
    #     '--pose-dist-cap-px',
    #     type=int,
    #     default=15,
    #     help='each pose keypoints distance value will be capped by this argument, So'
    #          ' distances greater than this cap will be set to the cap value.',
    # )
    # parser.add_argument(
    #     '--dist-out-file',
    #     help='append the mean average distance to this file',
    # )

    args = parser.parse_args()

    if args.image_out_dir is not None:
        os.makedirs(args.image_out_dir, exist_ok=True)

    print('=> loading configuration from {}'.format(args.cfg))

    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    if args.model_file:
        cfg.TEST.MODEL_FILE = args.model_file
    cfg.freeze()

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model_extra = cfg.MODEL.EXTRA

    with torch.no_grad():

        model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
            cfg, is_train=False
        )
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
        model.eval()
        model = model.cuda()

        normalize = transforms.Normalize(
            mean=[0.485], std=[0.229]
        )

        image_list_filename = args.image_list
        img_names = None
        if image_list_filename is not None:
            img_names = set()
            with open(image_list_filename) as val_file:
                for curr_line in val_file:
                    img_name = curr_line.strip()
                    img_names.add(img_name)

        fecal_boli_labels = list(itertools.chain.from_iterable(parse_fecal_boli_labels(f) for f in args.cvat_files))
        if img_names is not None:
            fecal_boli_labels = [l for l in fecal_boli_labels if l['image_name'] in img_names]

        for label in fecal_boli_labels:
            image_name = label['image_name']
            print('image_name:', image_name)
            # print([pi.keypoints for pi in label_pose_instances])

            image_path = os.path.join(args.image_dir, image_name)

            image_data_numpy = skimage.io.imread(image_path, as_gray=True)

            image_data = torch.from_numpy(image_data_numpy).to(torch.float32)
            image_data = normalize(image_data.unsqueeze(0)).squeeze(0)
            image_data = torch.stack([image_data] * 3)

            # add a size 1 batch dimension to the image and move it to the GPU
            image_data = image_data.unsqueeze(0).cuda()

            heatmaps = model(image_data)
            # pose_localmax = aeutil.localmax2D(heatmaps, args.min_pose_heatmap_val, 3)

            batch_index = 0
            # inferred_pose_instances = aeutil.calc_pose_instances(
            #     heatmaps[batch_index, ...],
            #     pose_localmax[batch_index, ...],
            #     inst_embed_data[batch_index, ...],
            #     min_embed_sep,
            #     max_embed_sep,
            #     max_inst_dist)

            # # filter out pose instances that have too few points
            # inferred_pose_instances = [
            #     p for p in inferred_pose_instances
            #     if len(p.keypoints) >= args.minimum_keypoint_count]
            # print('== {} POSE INSTANCES FROM INFERENCE =='.format(len(inferred_pose_instances)))
            # # print([pi.keypoints for pi in inferred_pose_instances])

            if args.image_out_dir is not None:
                image_rgb = np.zeros([image_data_numpy.shape[0], image_data_numpy.shape[1], 3], dtype=np.float32)
                image_rgb[...] = image_data_numpy[..., np.newaxis]

                # for pose_index, pose_instance in enumerate(inferred_pose_instances):
                #     for keypoint in pose_instance.keypoints.values():
                #         rr, cc = skimage.draw.circle(
                #             keypoint['y_pos'], keypoint['x_pos'],
                #             3,
                #             image_rgb.shape)
                #         skimage.draw.set_color(image_rgb, (rr, cc), colors[pose_index % len(colors)])

                if args.plot_heatmap:
                    _, axs = plt.subplots(1, 2, figsize=(12, 6))
                else:
                    _, axs = plt.subplots(1, 1, figsize=(6, 6))
                axs[0].imshow(image_rgb, aspect='equal')


                if args.plot_heatmap:
                    curr_heatmaps = heatmaps[batch_index, ...]
                    max_heatmap, _ = curr_heatmaps.max(dim=0)
                    max_heatmap_np = max_heatmap.cpu().numpy()
                    # max_heatmap_np[20, 20] = 1
                    axs[1].imshow(max_heatmap_np)

                image_base, _ = os.path.splitext(os.path.basename(image_name))
                plt.savefig(os.path.join(
                    args.image_out_dir,
                    image_base + '_fecal_boli_detect.png'))

                plt.close()

            # # match up poses by distance
            # pose_combos = []
            # for lbl_pose_i, lbl_pose in enumerate(label_pose_instances):
            #     for inf_pose_i, inf_pose in enumerate(inferred_pose_instances):
            #         curr_dist, _ = capped_pose_distance(inf_pose, lbl_pose, args.pose_dist_cap_px)
            #         pose_combos.append((lbl_pose_i, inf_pose_i, curr_dist))

            # # sort pose combinations by distance
            # pose_combos.sort(key=lambda pcombo: pcombo[2])

            # pose_dist_sum = 0
            # lbl_pose_count = len(label_pose_instances)
            # inf_pose_count = len(inferred_pose_instances)
            # unmatched_lbl_poses = set(range(lbl_pose_count))
            # unmatched_inf_poses = set(range(inf_pose_count))
            # for lbl_pose_i, inf_pose_i, curr_dist in pose_combos:
            #     if lbl_pose_i in unmatched_lbl_poses and inf_pose_i in unmatched_inf_poses:
            #         pose_dist_sum += curr_dist
            #         unmatched_lbl_poses.remove(lbl_pose_i)
            #         unmatched_inf_poses.remove(inf_pose_i)

            # # unmatched poses will be treated as if every point is at the capped distance
            # pose_count_diff = abs(inf_pose_count - lbl_pose_count)
            # pose_dist_sum += args.pose_dist_cap_px * pose_count_diff

            # max_pose_count = max(lbl_pose_count, inf_pose_count)
            # pose_dist_avg = pose_dist_sum / max_pose_count
            # pose_dist_avg_sum += pose_dist_avg

            # print('$$$$$$$$ POSE DIST:', pose_dist_avg)
            # print()

        # pose_dist_mean_avg = pose_dist_avg_sum / len(fecal_boli_labels)
        # print('pose_dist_mean_avg:', pose_dist_mean_avg)

        # if args.dist_out_file is not None:
        #     with open(args.dist_out_file, 'a') as dist_out_file:
        #         dist_out_file.write('{}\t{}\n'.format(args.cfg, pose_dist_mean_avg))


if __name__ == "__main__":
    main()
