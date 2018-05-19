from collections import defaultdict

import numpy as np

class DBSCAN:
    """ Simple implementation of DBSCAN clustering
    """
    def __init__(self, data, eps, minpts):
        self.data = data
        self.labels = [0] * len(data)

        self.eps = eps
        self.minpts = minpts
        self.count = 0
        self._clusters = defaultdict(list)

    def __len__(self):
        return self.count

    @property
    def clusters(self):
        if not self._clusters: self._dbscan()
        return self._clusters

    def _grow(self, index, neighbors):
        self.count += 1
        self.labels[index] = self.count
        for neighbor in neighbors:
            if self.labels[neighbor] == -1:
                self.labels[neighbor] = self.count
            elif not self.labels[neighbor]:
                self.labels[neighbor] = self.count
                sub_neighbors = self._neighbor(neighbor)
                if len(sub_neighbors) >= self.minpts:
                    neighbors += sub_neighbors

    def _neighbor(self, index):
        condition = lambda pos: np.linalg.norm(self.data[index] - pos) <= self.eps
        return [i for i, pos in enumerate(self.data) if condition(pos)]

    def _dbscan(self):
        for index, _ in enumerate(self.data):
            if self.labels[index]: continue
            neighbors = self._neighbor(index)
            if len(neighbors) < self.minpts:
                self.labels[index] = -1
            else:
                self._grow(index, neighbors)
        for index, label in enumerate(self.labels):
            self._clusters[label].append(index)
