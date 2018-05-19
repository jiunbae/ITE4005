# -*- coding: utf-8 -*-
"""Random forest

This is part of ITE4005 assignment#2 @ Hanyang Univ.
Author: Bae Jiun, Maybe

This module implement random forest
"""

from typing import List, Dict
from collections import Counter

import numpy as np
from numpy import apply_along_axis as apply

from .estimator import Estimator


class RandomForest(Estimator):
    """Random forest
    """

    def __init__(self, estimator: Estimator, size: int, *args, **kwargs):
        """Random forest with selected estimators

        :param estimator:       estimator for generate forest
        :param size:            forest size
        :param args, kwargs:    parameters pass through estimator
        """
        self.size = size
        assert issubclass(estimator, Estimator), 'estimator must be a child of Estimator'
        self.estimators = [estimator(*args, **kwargs) for _ in range(size)]

    def fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs) -> None:
        """fit random forest to input X, y

        @param: X               train X
        @param: y               train y
        :param args, kwargs:    parameters pass through estimatore
        """
        train_data = np.concatenate([np.concatenate([X, np.array([y]).T], axis=1)] * self.size)
        np.random.shuffle(train_data)

        for estimator, data in zip(self.estimators, np.array_split(train_data, self.size)):
            estimator.fit(data[:, :-1], data[:, -1], *args, **kwargs)

    def predict(self, X: np.ndarray) -> List[float]:
        """predict using generated random forest with input X

        @param: X               test X
        """
        pred = np.array([e.predict(X) for e in self.estimators])
        return apply(lambda y: Counter(y).most_common()[0][0], 1, pred.T)
