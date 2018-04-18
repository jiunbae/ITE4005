#!/bin/python3

import argparse
from collections import Counter

import pandas as pd
import numpy as np

from lib.metric import METRICS
from lib.tree import DecisionTree

def main(train_file, test_file, result_file, args):
    # load data
    train = pd.read_csv(train_file, sep='\t')
    test = pd.read_csv(test_file, sep='\t')

    # y_label, class to predict
    y_label, *_ = list(set(train.columns) - set(test.columns))

    # categorical data to numeric data (just encode)
    df, labels = pd.concat([train, test]), dict()
    for column in df:
        df[column], labels[column] = pd.factorize(df[column])
        labels[column] = labels[column].get_values()

    # split x, y, train, test set
    x_train = df.iloc[:len(train)].drop(y_label, axis=1)
    y_train = df.iloc[:len(train)][y_label]
    x_test = df.iloc[len(train):].drop(y_label, axis=1)

    # process single tree
    if args.forest == 1:
        classifier = DecisionTree(METRICS[args.metric], args.feature)

        # fit train data to classifier
        classifier.fit(x_train, y_train, args.depth, args.minsize, args.mingain)

        test[y_label] = pd.Series(map(lambda y: labels[y_label][y], classifier.predict(x_test)))
    # process random forest
    else:
        forest = [DecisionTree(METRICS[args.metric], args.feature) for _ in range(args.forest)]

        # duplicate data and shuffle
        X = np.concatenate([x_train, np.array([y_train]).T], axis=1)
        dataset = np.concatenate([X] * args.forest)
        np.random.shuffle(dataset)

        # fit train data to each tree
        for tree, data in zip(forest, np.array_split(dataset, args.forest)):
            tree.fit(data[:, :-1], data[:, -1], args.depth, args.minsize, args.mingain)

        pred = np.array([tree.predict(x_test) for tree in forest])
        test[y_label] = np.apply_along_axis(lambda y: labels[y_label][Counter(y).most_common()[0][0]], 1, pred.T)

    # write result
    test.to_csv(result_file, sep='\t', index=None)

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("train", help="minimum support", type=str)
    parser.add_argument("test", help="input file name", type=str)
    parser.add_argument("result", help="output file name", type=str)

    parser.add_argument("--metric", help="select metric to apply", dest='metric', choices=METRICS.keys(), default='entropy')
    parser.add_argument("--depth", help="maximum depth of tr1ee", dest='depth', type=int, default=32)
    parser.add_argument("--minsize", help="minimum size of node", dest='minsize', type=int, default=0)
    parser.add_argument("--mingain", help="minimum gain of node to divide", dest='mingain', type=float, default=.3)
    parser.add_argument("--feature", help="maximum feature count", dest='feature', type=int, default=0)
    parser.add_argument("--forest", help="size of forest", dest='forest', type=int, default=1)
    args = parser.parse_args()

    main(args.train, args.test, args.result, args)
