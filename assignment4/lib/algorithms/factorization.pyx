from functools import reduce
from typing import Any
from time import sleep

cimport numpy as np
import numpy as np

from lib.estimator import Estimator

class SVD(Estimator):
    def __init__(self, factors=100, epochs=20,
                 mean=0, derivation=.1, lr=.005,
                 reg=.02, random_state=None):
        self.state = random_state or np.random.RandomState()
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

        cdef list uniques
        cdef list biases
        cdef list params
        cdef list indexs

        cdef np.ndarray[np.long_t] u
        cdef np.ndarray[np.double_t] b
        cdef np.ndarray[np.double_t, ndim=2] p

        cdef int i, f
        cdef double rate, dot, err, v, t
        cdef np.ndarray[np.long_t] values

        cdef double lr = self.lr
        cdef double reg = self.reg
        cdef mean = self.__mean

        uniques = [np.unique(X[:, i]) for i in range(np.size(X, 1))]
        biases = [np.zeros(u.size, np.double) for u in uniques]
        params = [state.normal(self.mean, self.dev, (u.size, self.factors)) for u in uniques]

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
        predictions = np.full(np.size(X, 0), self.__mean)

        for i, xs in enumerate(X):
            known = [x in u for u, x in zip(self.uniques, xs)]
            predictions[i] += sum(b[x] for k, b, x in zip(known, self.biases, xs) if k)

        return predictions
