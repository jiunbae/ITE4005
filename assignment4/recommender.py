#!/bin/python3

import argparse

import pandas as pd

def main(train_file, test_file):
    data = pd.read_csv(train_file, sep='\t', header=None)

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
