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

    def fit(self, X, y):
        cdef list uniques
        cdef list biases
        cdef list params

        cdef np.ndarray[np.long_t] unique
        cdef np.ndarray[np.double_t, ndim=2] pu
        cdef np.ndarray[np.double_t, ndim=2] qi

        cdef int u, i, f
        cdef double r, err, dot, puf, qif
        cdef double mean = np.mean(y)

        cdef double lr = self.lr
        cdef double reg = self.reg

        uniques = [np.unique(X[:, i]) for i in range(np.size(X, 1))]
        biases = [np.zeros(unique.size, np.double) for unique in uniques]
        params = [self.state.normal(self.init_mean, self.init_dev,
                                   (unique.size, self.factors)) for unique in uniques]

        for _ in range(self.epochs):
            for (u, i), r in zip(X, y):
                u = np.where(uniques[0] == u)[0][0]
                i = np.where(uniques[1] == i)[0][0]

                dot = 0 
                for f in range(self.factors):
                    dot += params[1][i, f] * params[0][u, f]
                err = r - (mean + biases[0][u] + biases[1][i] + dot)

                biases[0][u] += lr * (err - reg * biases[0][u])
                biases[1][i] += lr * (err - reg * biases[1][i])

                for f in range(self.factors):
                    puf = params[0][u, f]
                    qif = params[1][i, f]
                    params[0][u, f] += lr * (err * qif - reg * puf)
                    params[1][i, f] += lr * (err * puf - reg * qif)

        self.mean = mean
        self.uniques = uniques
        self.biases = biases
        self.params = params

    def predict(self, X):
        estimate = np.full(np.size(X, 0), self.mean)
        for e, (u, i) in enumerate(X):
            known_user = u in self.uniques[0]
            known_item = i in self.uniques[1]

            if known_user:
                u = np.where(self.uniques[0] == u)[0][0]
                estimate[e] += self.biases[0][u]

            if known_item:
                i = np.where(self.uniques[1] == i)[0][0]
                estimate[e] += self.biases[1][i]

            if known_user and known_item:
                estimate[e] += np.dot(self.params[1][i], self.params[0][u])
        return estimate
