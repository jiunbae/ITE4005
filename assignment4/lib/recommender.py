# -*- coding: utf-8 -*-
"""Decision tree

This is part of ITE4005 assignment#2 @ Hanyang Univ.
Author: Bae Jiun, Maybe

This module implement decision tree
"""

from typing import Any

import numpy as np

from .algorithms import SVD

class Recommender:
    """Recommender
    """

    def __init__(self, algorithm=None, **kwargs):
        self.metric = (algorithm or SVD)(**kwargs)

    def fit(self, 
            X: np.ndarray,
            y: np.ndarray):
        self.metric.fit(X, y)

    def predict(self, X: np.ndarray) -> Any:
        return self.metric.predict(X)
