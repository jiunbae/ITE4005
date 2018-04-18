from collections import Counter
from functools import partial
from random import sample

import numpy as np

from .metric import Metric, Gini

class DecisionTree():
    '''Decision Tree
    '''

    def __init__(self, metric=Gini, max_feature=0):
        '''DecisionTree with selected metric

        @param: metric          metric for determin tree
        '''
        self.metric = metric().calc if issubclass(metric, Metric) else metric
        self.max_feature = max_feature

    def _build(self, train, depth=1):
        # split data where value lower than v and else
        _split = lambda i, v: (train[np.where(train[:, i] < v)], train[np.where(train[:, i] >= v)])
        # information gain
        _gain = lambda groups: sum(map(self.metric, groups))/len(groups)-self.metric(train)
        # _split and calc value using metric
        _apply = np.vectorize(lambda v, i: _gain(_split(i, v)))
        # get index of _applyed minimum
        _mini = lambda uni, idx, i: idx[_apply(uni, i).argmin()]
        # terminal node
        _terminal = lambda x: Counter(x[:, -1]).most_common()
        
        m = np.apply_along_axis(lambda i: _mini(*np.unique(train[:, i], return_index=True), i), 1, np.array([self.features]).T)
        i, j = min(enumerate(m), key=lambda t: _gain(_split(t[0], train[t[1]][t[0]])))
        l, r = _split(i, train[j][i])
        
        node = {
            'index': i,
            'value': train[j][i],
            'left': l,
            'right': r,
        }

        if not len(l) or not len(r):
            node['left'] = node['right'] = _terminal(np.concatenate([l, r]))
        elif depth >= self.max_depth:
            node['left'], node['right'] = _terminal(l), _terminal(r)
        else:
            node['left'] = _terminal(l) if len(l) <= self.min_size else self._build(l, depth+1)
            node['right'] = _terminal(r) if len(r) <= self.min_size else self._build(r, depth+1)

        return node
    
    def _predict(self, node, x):
        tar = node['left'] if x[node['index']] < node['value'] else node['right']
        return self._predict(tar, x) if isinstance(tar, dict) else tar[0][0]

    def fit(self, X, y, max_depth=32, min_size=0):
        '''fit decision tree to input X, y

        @param: X               train X
        @param: y               train y
        @param: max_depth(32)   maximum tree depth
        @param: min_size(0)     minimum size to become a leaf node
        '''
        self.max_depth = max_depth
        self.min_size = min_size
        
        train = np.concatenate([X, np.array([y]).T], axis=1)
        self._r, self._c = train.shape
        self.features = sorted(sample(range(self._c - 1), self.max_feature if self.max_feature else self._c-1))
        self.tree = self._build(train)

    def predict(self, X):
        '''predict using generated decision tree with input X

        @param: X               test X
        '''
        return np.apply_along_axis(partial(self._predict, self.tree), 1, X)
