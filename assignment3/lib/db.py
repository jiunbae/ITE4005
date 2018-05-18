from collections import defaultdict

import numpy as np

class DBSCAN:
    def __init__(self, data, eps, minpts):
        self.obj = data[:, 0]
        self.pos = data[:, 1:]
        self.labels = [0] * len(data)

        self.eps = eps
        self.minpts = minpts
        self.count = 0
        self.clusters = defaultdict(list)
        
    def __len__(self):
        return self.count

    def _metric(self, left, right):
        return np.linalg.norm(left - right)
    
    def _grow(self, index, neighbors):
        self.labels[index] = self.count
        for neighbor, *_ in neighbors:
            if self.labels[neighbor] == -1:
                self.labels[neighbor] = self.count
            elif not self.labels[neighbor]:
                self.labels[neighbor] = self.count
                sub_neighbors = self._neighbor(neighbor)
                if len(sub_neighbors) >= self.minpts:
                    neighbors += sub_neighbors

    def _neighbor(self, index):
        condition = lambda x: self._metric(self.pos[index], x) <= self.eps
        return [(i, *pos) for i, pos in enumerate(self.pos) if condition(pos)]
        
    def _dbscan(self):
        for index, _ in enumerate(self.obj):
            if self.labels[index]: continue
            neighbors = self._neighbor(index)
            if len(neighbors) < self.minpts:
                self.labels[index] = -1
            else:
                self.count += 1
                self._grow(index, neighbors)
        for obj, label in zip(self.obj, self.labels):
            self.clusters[label].append(obj)

    def get(self, n=0):
        if not self.clusters: self._dbscan()
        if not n: return self.clusters.items()
        if n > len(self.clusters):
            raise Exception('Not enough clusters')
        clusters = defaultdict(list)
        for key, value in sorted(self.clusters.items(), key=lambda x: -len(x[1])):
            if len(clusters) == n: break
            if key != -1:
                clusters[key] = value
        return clusters.items()
