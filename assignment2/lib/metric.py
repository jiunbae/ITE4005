# -*- coding: utf-8 -*-
"""Decision tree metrics

This is part of ITE4005 assignment#2 @ Hanyang Univ.
Author: Bae Jiun, Maybe

This module implement metrics for decision tree.
"""

from typing import List, Iterator, Tuple, Any
from collections import Counter

import numpy as np


class Metric:
    """Default Metric class to measure impurity degree

    **DO NOT USE THIS RIGHT AWAY**
    Inherit Metric and implement the `calc` function
    """
    def __init__(self):
        pass

    def p(self, g: List[np.ndarray], count: float) -> float:
        """calculate probability of class

        :param g:       input group
        :param count:   count of class

        :return probability of class in group
        """
        return count / len(g)

    def classes(self, g: List[np.ndarray]) -> Iterator[Tuple[Any, int]]:
        """classes return generator of class and count in group
        """
        for key, count in Counter(g[:, -1]).most_common():
            yield key, count

    def calc(self, g: List[np.ndarray]) -> float:
        """return calculated value using metric
        """
        return sum([self.p(g, v) for _, v in self.classes(g)])


class Entropy(Metric):
    """Entropy

    Sum(j) -p[j] * log2(p[j])
    """
    def calc(self, g: List[np.ndarray]) -> float:
        return -sum([self.p(g, v)*np.log(self.p(g, v))/np.log(2) for _, v in self.classes(g)])


class Gini(Metric):
    """Gini

    1 - Sum(j) p[j]^2
    """
    def calc(self, g: List[np.ndarray]) -> float:
        return 1.-sum([(self.p(g, v))**2 for _, v in self.classes(g)])


class ClassError(Metric):
    """Classification error

    1 - max{P[j]}
    """
    def calc(self, g: List[np.ndarray]) -> float:
        return 1.-next(self.classes(g))[1]/len(g) if g.size else 1


METRICS = {
    'entropy': Entropy,
    'error': ClassError,
    'gini': Gini,
}
