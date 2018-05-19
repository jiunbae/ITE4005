# -*- coding: utf-8 -*-
"""Estimator

This is part of ITE4005 assignment#2 @ Hanyang Univ.
Author: Bae Jiun, Maybe

This module for definition of estimator
"""

from typing import List

class Estimator:
    """Base estimator
    """

    def __init__(self):
        """Base estimator
        """
        pass

    def fit(self, X: List, y: List) -> None:
        """fit estimator to input X, y

        @param: X               train X
        @param: y               train y
        """
        pass

    def predict(self, X: List[List]) -> List[float]:
        """predict using trained estimator with input X

        @param: X               test X
        """
        pass
