#!/bin/python3

import argparse

import pandas as pd

from lib.metric import Entropy, ClassError, Gini
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

    classifier = DecisionTree({
        'entropy': Entropy,
        'error': ClassError,
        'gini': Gini,
    }[args.metric], args.feature)

    classifier.fit(x_train, y_train, args.depth, args.minsize, args.mingain)

    test[y_label] = pd.Series(map(lambda y: labels[y_label][y], classifier.predict(x_test)))

    # write result
    test.to_csv(result_file, sep='\t', index=None)

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("train", help="minimum support", type=str)
    parser.add_argument("test", help="input file name", type=str)
    parser.add_argument("result", help="output file name", type=str)

    parser.add_argument("--metric", help="select metric to apply", dest='metric', choices=['entropy','error','gini'], default='entropy')
    parser.add_argument("--depth", help="maximum depth of tr1ee", dest='depth', type=int, default=32)
    parser.add_argument("--minsize", help="minimum size of node", dest='minsize', type=int, default=0)
    parser.add_argument("--mingain", help="minimum gain of node to divide", dest='mingain', type=float, default=.3)
    parser.add_argument("--feature", help="maximum feature count", dest='feature', type=int, default=0)
    args = parser.parse_args()

    main(args.train, args.test, args.result, args)
