from functools import reduce

import numpy as np
cimport numpy as np

from lib.estimator import Estimator

class SVD:
    def __init__(self, factors=100, epochs=20,
                 mean=.0, derivation=.1, lr=.005,
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
        cdef list uniques
        cdef list biases
        cdef list params

        cdef np.ndarray[np.long_t] unique_user, unique_item
        cdef np.ndarray[np.double_t] biase_user, biase_item
        cdef np.ndarray[np.double_t, ndim=2] param_user, param_item

        cdef int u, i, f
        cdef double r, err, dot, param_userf, param_itemf
        cdef double mean = np.mean(y)

        cdef double lr = self.lr
        cdef double reg = self.reg

        unique_user = np.unique(X[:, 0])
        unique_item = np.unique(X[:, 1])
        biase_user = np.zeros(unique_user.size, np.double)
        biase_item = np.zeros(unique_item.size, np.double)
        param_user = self.state.normal(self.init_mean, self.init_dev,
                                      (unique_user.size, self.factors))
        param_item = self.state.normal(self.init_mean, self.init_dev,
                                      (unique_item.size, self.factors))

        for _ in range(self.epochs):
            for (u, i), r in zip(X, y):
                u = np.where(unique_user == u)[0][0]
                i = np.where(unique_item == i)[0][0]

                dot = 0 
                for f in range(self.factors):
                    dot += param_item[i, f] * param_user[u, f]
                err = r - (mean + biase_user[u] + biase_item[i] + dot)

                biase_user[u] += lr * (err - reg * biase_user[u])
                biase_item[i] += lr * (err - reg * biase_item[i])

                for f in range(self.factors):
                    param_user[u, f] += lr * (err * param_item[i, f] - reg * param_user[u, f])
                    param_item[i, f] += lr * (err * param_user[u, f] - reg * param_item[i, f])

        self.mean = mean

        self.unique_user = unique_user
        self.unique_item = unique_item
        self.biase_user = biase_user
        self.biase_item = biase_item
        self.param_user = param_user
        self.param_item = param_item

    def predict(self, X: np.ndarray) -> np.ndarray:
        cdef np.ndarray[np.double_t] estimate
        cdef double mean = self.mean

        estimate = np.full(np.size(X, 0), mean)
        for e, (u, i) in enumerate(X):
            known_user = u in self.unique_user
            known_item = i in self.unique_item

            if known_user:
                u = np.where(self.unique_user == u)[0][0]
                estimate[e] += self.biase_user[u]

            if known_item:
                i = np.where(self.unique_item == i)[0][0]
                estimate[e] += self.biase_item[i]

            if known_user and known_item:
                estimate[e] += np.dot(self.param_item[i], self.param_user[u])
        return estimate
