#!/bin/python3
# -*- coding: utf-8 -*-

"""Recommender System

This is part of ITE4005 assignment#4 @ Hanyang Univ.
Author: Bae Jiun, Maybe
"""
import argparse
from os.path import splitext
from time import time
import pandas as pd

from lib import timer
from lib.recommender import Recommender

def main(train_file, test_file, args):
    timer.begin()
    train = pd.read_csv(train_file, sep='\t', header=None)
    test = pd.read_csv(test_file, sep='\t', header=None)

    model = Recommender(**args)
    model.fit(train.values[:, :2], train.values[:, 2])

    predicts = model.predict(test.values[:, :2])
    test = test.drop(test.columns[-1], axis=1)
    test[test.columns[-1]] = pd.Series(predicts)
    test.to_csv(splitext(test_file)[0] + '.base_prediction.txt', sep='\t', index=None, header=None)
    print ('recommender performed in', timer.end())

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("train",
                        help="train file path",
                        type=str)
    parser.add_argument("test",
                        help="test file path",
                        type=str)

    parser.add_argument("--factor",
                        dest="factor", help="size of factor", type=int, default=100)
    parser.add_argument("--epoch",
                        dest="epoch", help="epoch size", type=int, default=20)
    parser.add_argument("--mean",
                        dest="mean", help="initial mean value", type=float, default=.0)
    parser.add_argument("--dev",
                        dest="dev", help="initial derivation value", type=float, default=.1)
    parser.add_argument("--lr",
                        dest="lr", help="leraning rate", type=float, default=.005)
    parser.add_argument("--reg",
                        dest="reg", help="regression rate", type=float, default=.02)

    args = parser.parse_args()
    main(args.train, args.test, {
        'factors': args.factor,
        'epochs': args.epoch,
        'mean': args.mean,
        'derivation': args.dev,
        'lr': args.lr,
        'reg': args.reg,
    })
