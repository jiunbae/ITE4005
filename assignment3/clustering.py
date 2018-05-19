#!/bin/python3

import argparse
from os import path

import pandas as pd
import numpy as np

from lib.clustering import DBSCAN

def main(input_file, n, eps, min_pts):
    data = pd.read_csv(input_file, sep='\t', header=None)

    model = DBSCAN(data.values, eps, min_pts)

    base = path.splitext(input_file)[0]

    for key, values in model.get(n):
        with open(base + '_cluster_' + str(key) + '.txt', 'w') as f:
            for value in values:
                f.write(str(int(value)) + '\n')

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input",
                        help="input file name",
                        type=str)
    parser.add_argument("n",
                        help="number of clusters for the corresponding input data",
                        type=int)
    parser.add_argument("eps",
                        help="maximum radius of the neighborhood",
                        type=int)
    parser.add_argument("min",
                        help="minimum number of points in an Eps-neighborhood of a given point",
                        type=int)

    args = parser.parse_args()

    main(args.input, args.n, args.eps, args.min)
