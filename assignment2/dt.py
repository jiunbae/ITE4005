#!/bin/python3
# -*- coding: utf-8 -*-

"""Decision tree assignment

This is part of ITE4005 assignment#2 @ Hanyang Univ.
Author: Bae Jiun, Maybe
"""

import argparse
from collections import Counter

import pandas as pd
import numpy as np
from numpy import apply_along_axis as apply

from lib.metric import METRICS
from lib.tree import DecisionTree
from lib.forest import RandomForest


def main(train_file:str, test_file: str, result_file: str, args: argparse.Namespace) -> None:
    # load data
    train = pd.read_csv(train_file, sep='\t')
    test = pd.read_csv(test_file, sep='\t')

    # label, class to predict
    label, *_ = list(set(train.columns) - set(test.columns))

    # categorical data to numeric data (just encode)
    dataframe, labels = pd.concat([train, test]), dict()
    for column in dataframe:
        dataframe[column], labels[column] = pd.factorize(dataframe[column])
        labels[column] = labels[column].get_values()

    # split x, y, train, test set
    x_train = dataframe.iloc[:len(train)].drop(label, axis=1)
    y_train = dataframe.iloc[:len(train)][label]
    x_test = dataframe.iloc[len(train):].drop(label, axis=1)

    classifier = DecisionTree(METRICS[args.metric], args.feature) 
    if args.forest > 1:
        calssifier = RandomForest(DecisionTree, args.forest, METRICS[args.metric], args.feature)

    # fit train data to classifier
    classifier.fit(x_train, y_train, args.depth, args.minsize, args.mingain)

    test[label] = pd.Series(map(lambda y: labels[label][y], classifier.predict(x_test)))

    # write result
    test.to_csv(result_file, sep='\t', index=None)


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("train", help="train data file path", type=str)
    parser.add_argument("test", help="test data file path", type=str)
    parser.add_argument("result", help="output file path", type=str)

    parser.add_argument("--metric", help="select metric to apply",
                        dest='metric', choices=METRICS.keys(), default='entropy')
    parser.add_argument("--depth", help="maximum depth of tr1ee",
                        dest='depth', type=int, default=32)
    parser.add_argument("--minsize", help="minimum size of node",
                        dest='minsize', type=int, default=0)
    parser.add_argument("--mingain", help="minimum gain of node to divide",
                        dest='mingain', type=float, default=.3)
    parser.add_argument("--feature", help="maximum feature count",
                        dest='feature', type=int, default=0)
    parser.add_argument("--forest", help="size of forest",
                        dest='forest', type=int, default=1)
    args = parser.parse_args()

    main(args.train, args.test, args.result, args)
