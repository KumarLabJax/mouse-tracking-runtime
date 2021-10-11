import argparse
import re

import csv
import functools
import itertools
import os
import pprint
import random
import shutil

import _init_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description='find the given netids in a list of fully quallified paths',
    )

    parser.add_argument(
        'netids',
        help='file containing newline separated netids',
    )
    parser.add_argument(
        'fullpaths',
        help='file containing full paths to video files',
    )
    parser.add_argument(
        '--print-missing',
        action='store_true',
    )
    parser.add_argument(
        '--rename-map',
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    print('args.rename_map:', args.rename_map)

    with open(args.fullpaths, 'r') as fullpaths_file, open(args.netids) as netids_file:
        fullpaths = [p.strip() for p in fullpaths_file]
        for netid in netids_file:
            netid = netid.strip()

            matching_fullpath = None
            for p in fullpaths:
                if p.endswith(netid):
                    matching_fullpath = p
                    break
            
            if matching_fullpath is None and args.print_missing:
                print(netid)
            elif matching_fullpath is not None and not args.print_missing:
                print(matching_fullpath)

if __name__ == '__main__':
    main()
