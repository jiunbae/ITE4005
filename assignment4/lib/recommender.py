# -*- coding: utf-8 -*-
"""Decision tree

This is part of ITE4005 assignment#2 @ Hanyang Univ.
Author: Bae Jiun, Maybe

This module implement decision tree
"""
from typing import Any

import numpy as np

from .algorithms import SVD

class Recommander:
    def __init__(self, algorithm=SVD()):
        pass

    def fit(self, 
            X: np.ndarray,
            y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> Any:
        pass