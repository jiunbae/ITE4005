#!/bin/python3
# -*- coding: utf-8 -*-

"""Recommender System

This is part of ITE4005 assignment#4 @ Hanyang Univ.
Author: Bae Jiun, Maybe
"""
import argparse
from os.path import splitext

import pandas as pd

from .lib.recommender import Recommender

def main(train_file, test_file):
    train = pd.read_csv(train_file, sep='\t', header=None)
    test = pd.read_csv(test_file, sep='\t', header=None)

    model = Recommender()
    model.fit(train.values[:, :2], train.values[:, -1])

    test['rating'] = pd.Series(model.predict(test.values))
    test.to_csv(splitext(test_file)[0] + '.base_prediction.txt', sep='\t', index=None)

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("train",
                        help="train file path",
                        type=str)
    parser.add_argument("test",
                        help="test file path",
                        type=str)

    args = parser.parse_args()
    main(args.train, args.test)
