from functools import reduce
from typing import Any
from time import sleep

import numpy as np

from lib.estimator import Estimator

class SVD(Estimator):
    def __init__(self, factors=100, epochs=20,
                 mean=0., derivation=.1, lr=.005,
                 reg=.02, random_state=None):
        self.state = random_state or np.random.RandomState(10)
        self.factors = factors
        self.epochs = epochs
        self.mean = mean
        self.dev = derivation
        self.lr = lr
        self.reg = reg

        self.uniques = None
        self.biases = None
        self.params = None

    def fit(self,
            X: np.ndarray,
            y: np.ndarray):
        state = self.state
        self.__mean = np.mean(y)

        uniques = [np.unique(X[:, i]) for i in range(np.size(X, 1))]
        biases = [np.zeros(u.size, np.double) for u in uniques]
        params = [state.normal(self.mean, self.dev, (u.size, self.factors)) for u in uniques]

        for _ in range(self.epochs):
            for values, rate in zip(X, y):
                indexs = [np.where(u == v)[0][0] for u, v in zip(uniques, values)]
                
                dot = sum(reduce(float.__mul__, [p[i, f] for p, i in zip(params, indexs)]) for f in range(self.factors))

                err = rate - (self.__mean + dot + sum(b[i] for b, i in zip(biases, indexs)))

                for b, i in zip(biases, indexs):
                    b[i] += self.lr * (err - self.reg * b[i])

                for f in range(self.factors):
                    t = sum(p[i, f] for p, i in zip(params, indexs))
                    for p, i in zip(params, indexs):
                        p[i, f] += self.lr * (err * (t - p[i, f])/(len(indexs)-1) - self.reg * p[i, f])

        self.uniques = uniques
        self.biases = biases
        self.params = params

    def predict(self, X: np.ndarray) -> Any:
        estimate = np.full(np.size(X, 0), self.__mean)

        for i, xs in enumerate(X):
            knowns = [x in u for u, x in zip(self.uniques, xs)]
            estimate[i] += sum(\
                            b[x] for b, k, x in zip(self.biases, knowns, xs) if k) +\
                           sum(np.multiply.reduce(\
                            [p[x] for p, k, x in zip(self.params, knowns, xs) if k])) *\
                             (np.size(X, 1)-1)
        return estimate
