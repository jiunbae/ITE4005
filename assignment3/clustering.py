#!/bin/python3

import argparse
from os import path

import pandas as pd

from lib.clustering import DBSCAN
from lib.plot import cluster_plot
from lib.timer import begin, end

def main(input_file, output_path, n, eps, min_pts, image):
    # Load data
    data = pd.read_csv(input_file, sep='\t', header=None).values
    base = path.splitext(path.basename(input_file))[0]
    index, values = data[:, 0], data[:, 1:]

    # DBSCAN
    begin()
    model = DBSCAN(values, eps, min_pts)
    clusters = model.clusters
    time = end()

    print (data.shape, 'input performed in', time)
    print (n, '/', len(clusters), 'clustered')

    del clusters[-1] # delete outliers

    # Write only top-n clusters
    for i, (_, objects) in enumerate(sorted(clusters.items(), key=lambda x: -len(x[1]))[:n]):
        with open(path.join(output_path, base) + '_cluster_' + str(i) + '.txt', 'w') as file:
            file.write('\n'.join([str(int(index[obj])) for obj in objects]))

    # Save cluster image
    if image:
        xs = data[:, 1]
        ys = data[:, 2]
        cs = model.labels
        cluster_plot(path.join(output_path, base + '.png'), xs, ys, cs)

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

    parser.add_argument("--output", dest="output", help="output file path", type=str, default="")
    parser.add_argument("--image", dest="image", help="save cluster image", action='store_true')

    args = parser.parse_args()
    output = args.output or path.dirname(args.input)
    main(args.input, output, args.n, args.eps, args.min, args.image)
