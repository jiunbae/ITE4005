#!/bin/python3

import argparse

import pandas as pd

from lib.metric import InformationGain, GainRatio, Gini
from lib.tree import DecisionTree

def main(train_file, test_file, result_file, metric):
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

    classifier = DecisionTree(metric)

    classifier.fit(x_train, y_train)

    test[y_label] = pd.Series(map(lambda y: labels[y_label][y], classifier.predict(x_test)))

    # write result
    test.to_csv(result_file, sep='\t', index=None)

if __name__ == '__main__':
    metrics = {
        'infogain': InformationGain,
        'gain': GainRatio,
        'gini': Gini,
    }

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("train", help="minimum support", type=str)
    parser.add_argument("test", help="input file name", type=str)
    parser.add_argument("result", help="output file name", type=str)
    parser.add_argument("--metric", help="select metric to apply", dest='metric', choices=metrics.keys())
    args = parser.parse_args()

    main(args.train, args.test, args.result, metrics.get(args.metric, Gini))
