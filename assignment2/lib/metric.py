from itertools import chain
from collections import Counter

import numpy as np

class Metric:
    def __init__(self):
        pass

    def calc(self, group):
        return 0

class Entropy(Metric):
    def calc(self, group):
        return -sum([v/len(group)*np.log(v/len(group))/np.log(2) for v in Counter(group[:, -1]).values()])

class ClassError(Metric):
    def calc(self, group):
        if not len(group): return 1
        return 1.-Counter(group[:, -1]).most_common()[0][1]/len(group)

class Gini(Metric):
    def calc(self, group):
        return 1.-sum([(v/len(group))**2 for v in Counter(group[:, -1]).values()])

METRICS = {
    'entropy': Entropy,
    'error': ClassError,
    'gini': Gini,
}
