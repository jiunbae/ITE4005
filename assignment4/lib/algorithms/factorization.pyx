from functools import reduce
from typing import Any
from time import sleep

import numpy as np
cimport numpy as np

from lib.estimator import Estimator

class SVD(Estimator):
    def __init__(self, factors=100, epochs=20,
                 mean=0., derivation=.1, lr=.005,
                 reg=.02, random_state=None):
        self.state = random_state or np.random.mtrand._rand
        self.factors = factors
        self.epochs = epochs
        self.init_mean = mean
        self.init_dev = derivation
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

        cdef list uniques
        cdef list biases
        cdef list params
        cdef list indexs

        # cdef np.ndarray[np.long_t] u
        # cdef np.ndarray[np.double_t] b
        # cdef np.ndarray[np.double_t, ndim=2] p

        # cdef int i, f
        cdef double rate, dot, err, v, t
        cdef np.ndarray[np.long_t] values

        cdef double lr = self.lr
        cdef double reg = self.reg
        cdef double mean = self.__mean

        uniques = [np.unique(X[:, i]) for i in range(np.size(X, 1))]
        biases = [np.zeros(u.size, np.double) for u in uniques]
        params = [state.normal(self.init_mean, self.init_dev, (u.size, self.factors)) for u in uniques]

        for _ in range(self.epochs):
            for values, rate in zip(X, y):
                indexs = [np.where(u == v)[0][0] for u, v in zip(uniques, values)]
                
                dot = sum(reduce(float.__mul__, [p[i, f] for p, i in zip(params, indexs)]) for f in range(self.factors))
                err = rate - mean - dot - sum(b[i] for b, i in zip(biases, indexs))

                for b, i in zip(biases, indexs):
                    b[i] += lr * (err - reg * b[i])

                for f in range(self.factors):
                    t = sum(p[i, f] for p, i in zip(params, indexs))
                    for p, i in zip(params, indexs):
                        p[i, f] += lr * (err * (t - p[i, f])/(len(indexs)-1) - reg * p[i, f])

        self.uniques = uniques
        self.biases = biases
        self.params = params

    def predict(self, X: np.ndarray) -> Any:
        cdef np.ndarray[np.long_t] u
        cdef np.ndarray[np.double_t] b
        cdef np.ndarray[np.double_t, ndim=2] p
        cdef np.ndarray[np.double_t] estimate

        cdef int i, x
        cdef bint k
        cdef double mean = self.__mean
        cdef list uniques = self.uniques
        cdef list biases = self.biases
        cdef list params = self.params
        cdef list knowns
        cdef list indexs

        estimate = np.full(np.size(X, 0), mean)

        for i, xs in enumerate(X):
            knowns = [x in u for u, x in zip(uniques, xs)]
            indexs = [(np.where(u == x)[0][0] if k else 0) for u, k, x in zip(uniques, knowns, xs)]

            estimate[i] += sum(\
                            b[x] for b, k, x in zip(biases, knowns, xs) if k) +\
                           sum(np.multiply.reduce(\
                            [p[x] for p, k, x in zip(params, knowns, xs) if k])) *\
                             (np.size(X, 1)-1)

        return estimate
