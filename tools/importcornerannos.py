"""
Convert CVAT corner annotations to an HDF annotation file
"""

import argparse
import h5py
import imageio
import numpy as np
import os
import random
import xml.etree.ElementTree as ET


# example(s):
# python tools/concatmousepose.py \
#   --cvat-anno in data/hdf5mouse/CCF1_Annotations.xml data/hdf5mouse/CCF1_Validation.txt data/hdf5mouse/CCF1_TrainingFrames \
#   --hdf5-out data/hdf5mouse/merged_pose_annos_2019-06-18.h5

def main():
    parser = argparse.ArgumentParser(
        description='Convert CVAT annotations to an HDF annotation file')

    # parser.add_argument(
    #  '--hdf5-in',
    #    help='input HDF5 file to concatenate to the output file',
    #    type=str,
    #    default=[],
    #    action='append',
    # )

    # parser.add_argument(
    #     '--cvat-val-fraction',
    #     help='fraction of CVAT samples to throw into the validation set',
    #     type=float,
    #     default=0.1,
    # )

    parser.add_argument(
        '--cvat-anno-in',
        help='CVAT XML file followed by newline separated list of validation images and image directory',
        type=str,
        nargs=3,
        default=[],
        action='append',
    )

    parser.add_argument(
        '--hdf5-out',
        help='HDF5 output file',
        required=True,
        type=str,
    )

    args = parser.parse_args()
    print(args)

    # all_cvat_annos = []
    val_cvat_annos = []
    train_cvat_annos = []

    for cvat_xml, cvat_val_list, img_dir in args.cvat_anno_in:

        cvat_val_set = set((
            line.strip()
            for line in open(cvat_val_list)
            if line.strip()
        ))

        tree = ET.parse(cvat_xml)
        root = tree.getroot()
        for image_elem in root.findall('./image'):
            # print(image_elem.attrib['name'])
            img_name = image_elem.attrib['name']
            frame_name, _ = os.path.splitext(img_name)
            image_path = os.path.join(img_dir, img_name)
            mouse_polylines = [
                pl for pl in image_elem.findall('./polyline')
                if pl.attrib['label'] == 'corner'
            ]

            if len(mouse_polylines) == 1:
                mouse_polyline = mouse_polylines[0]
                xy_strs = [xy_str.split(',') for xy_str in mouse_polyline.attrib['points'].split(';')]
                if len(xy_strs) == 4:
                    xy_points = [(round(float(x_str)), round(float(y_str))) for x_str, y_str in xy_strs]
                    # print(xy_points)
                    if img_name in cvat_val_set:
                        val_cvat_annos.append({
                            'frame_name': frame_name,
                            'image_path': image_path,
                            'points': xy_points,
                        })
                    else:
                        train_cvat_annos.append({
                            'frame_name': frame_name,
                            'image_path': image_path,
                            'points': xy_points,
                        })
                else:
                    print('BAD XY COUNT:', image_elem.attrib['name'], len(xy_strs))
            else:
                print('BAD MOUSE POLYLINE COUNT:', image_elem.attrib['name'], len(mouse_polylines))

    # random.shuffle(all_cvat_annos)
    # val_count = int(round(len(all_cvat_annos) * args.cvat_val_fraction))
    # val_cvat_annos = all_cvat_annos[:val_count]
    # train_cvat_annos = all_cvat_annos[val_count:]

    with h5py.File(args.hdf5_out, 'w') as hdf5_out:

        for ds_name, ds_annos in [('training', train_cvat_annos), ('validation', val_cvat_annos)]:
            for anno_dict in ds_annos:
                frame_name = anno_dict['frame_name']
                anno_grp = hdf5_out.create_group(ds_name + '/' + frame_name)

                image_arr = imageio.imread(anno_dict['image_path'], as_gray=True)
                image_arr = image_arr.astype(np.uint8)
                assert image_arr.ndim == 2
                anno_grp['frames'] = image_arr[np.newaxis, ..., np.newaxis]

                points = np.array(anno_dict['points'], dtype=np.uint16)
                anno_grp['points'] = points[np.newaxis, :]


if __name__ == "__main__":
    main()
