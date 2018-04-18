from itertools import chain
from collections import Counter

import numpy as np

class Metric:
    def __init__(self):
        pass

class Gini(Metric):
    def calc(self, groups):
        return sum([(1.-sum([(v/len(g))**2 for k, v in Counter(g[:, -1]).items()])) * (len(g)/len(list(chain(*groups)))) for g in filter(np.any, groups)])

class InformationGain(Metric):
    def calc(self, x):
        pass

class GainRatio(Metric):
    def calc(self, x):
        pass
