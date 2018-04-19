# -*- coding: utf-8 -*-
"""Decision tree

This is part of ITE4005 assignment#2 @ Hanyang Univ.
Author: Bae Jiun, Maybe

This module implement decision tree
"""

from collections import Counter
from functools import partial
from itertools import chain
from random import sample

import numpy as np
from numpy import apply_along_axis as apply

from .metric import Metric, Gini


class DecisionTree():
    """Decision Tree
    """

    def __init__(self,
                 metric: Metric = Gini,
                 max_feature: int = 0):
        """DecisionTree with selected metric

        :param metric:          metric for determine tree
        :param max_feature:     maximum feature size
        """
        self.metric = metric().calc if issubclass(metric, Metric) else metric
        self.max_feature = max_feature

        self.tree = None
        self.features = list()
        self.max_depth = 32
        self.min_gain = 0
        self.min_size = .005

    def _build(self, train: np.ndarray, depth: int = 1) -> dict:
        # split data where value lower than v and else
        _split = lambda i, v: (train[np.where(train[:, i] < v)], train[np.where(train[:, i] >= v)])
        # information gain
        _gain = lambda gs: -sum([self.metric(g) * len(g) / len(list(chain(*gs))) for g in gs])
        # _split and calc value using metric
        _apply = np.vectorize(lambda v, i: _gain(_split(i, v)))
        # get index of applied minimum
        _mini = lambda uni, idx, i: idx[_apply(uni, i).argmax()]
        # terminal node
        _t = lambda x: Counter(x[:, -1]).most_common()

        m = apply(lambda i: _mini(*np.unique(train[:, i], True), i), 1, np.array([self.features]).T)
        idx, row = max(zip(self.features, m), key=lambda t: _gain(_split(t[0], train[t[1]][t[0]])))
        left, right = _split(idx, train[row][idx])

        node = {
            'index': idx,
            'value': train[row][idx],
            'left': left,
            'right': right,
        }

        if not left.size or not right.size:
            node['left'] = node['right'] = _t(np.concatenate([left, right]))
        elif depth >= self.max_depth or -_gain([left, right]) < self.min_gain:
            node['left'], node['right'] = _t(left), _t(right)
        else:
            node['left'] = _t(left) if len(left) <= self.min_size else self._build(left, depth+1)
            node['right'] = _t(right) if len(right) <= self.min_size else self._build(right, depth+1)

        return node

    def _predict(self, node: dict, x: np.ndarray) -> float:
        tar = node['left'] if x[node['index']] < node['value'] else node['right']
        return self._predict(tar, x) if 'index' in tar else tar[0][0]

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            max_depth: int = 32,
            min_size: int = 0,
            min_gain: float = .005) -> None:
        """fit decision tree to input X, y

        @param: X               train X
        @param: y               train y
        @param: max_depth(32)   maximum tree depth
        @param: min_size(0)     minimum size to become a leaf node
        @param: min_gain(.005)  minimum information gain for create new node
        """
        self.max_depth = max_depth
        self.min_size = min_size
        self.min_gain = min_gain

        train = np.concatenate([X, np.array([y]).T], axis=1)
        self.features = sample(range(train.shape[1]-1), self.max_feature or train.shape[1]-1)
        # ordered feature number
        self.features = sorted(self.features)
        self.tree = self._build(train)

    def predict(self, X: np.ndarray) -> float:
        """predict using generated decision tree with input X

        @param: X               test X
        """
        return apply(partial(self._predict, self.tree), 1, X)
