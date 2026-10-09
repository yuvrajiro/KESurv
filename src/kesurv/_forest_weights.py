"""Bootstrap-aware survival forest proximity weights (paper Eq. 3)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


class ForestWeights:
    def __init__(self, forest, X_train):
        if not callable(getattr(forest, "apply", None)) or not hasattr(
            forest, "estimators_samples_"
        ):
            raise TypeError("forest must expose apply(X) and estimators_samples_")
        leaves = np.asarray(forest.apply(X_train))
        if leaves.ndim != 2 or leaves.shape[0] != len(X_train):
            raise ValueError("forest.apply(X_train) returned an invalid shape")
        self.n_train = len(X_train)
        self.n_trees = leaves.shape[1]
        if self.n_trees == 0:
            raise ValueError("forest contains no trees")
        # This property reconstructs bootstrap samples on every access.
        samples = forest.estimators_samples_
        if len(samples) != self.n_trees:
            raise ValueError("forest bootstrap samples and trees do not match")
        self.forest = forest
        self.leaf_maps = []
        for tree, draw in enumerate(samples):
            ids, counts = np.unique(np.asarray(draw, dtype=np.intp), return_counts=True)
            if len(ids) == 0 or ids[0] < 0 or ids[-1] >= self.n_train:
                raise ValueError("forest bootstrap samples contain invalid indices")
            groups = defaultdict(list)
            for index, multiplicity in zip(ids, counts):
                groups[int(leaves[index, tree])].append((int(index), int(multiplicity)))
            leaf_map = {}
            for leaf, pairs in groups.items():
                indices = np.fromiter((p[0] for p in pairs), dtype=np.intp)
                masses = np.fromiter((p[1] for p in pairs), dtype=np.float64)
                leaf_map[leaf] = (indices, masses / masses.sum())
            self.leaf_maps.append(leaf_map)

    def one(self, query_leaves):
        weights = np.zeros(self.n_train, dtype=np.float64)
        for tree, leaf in enumerate(query_leaves):
            entry = self.leaf_maps[tree].get(int(leaf))
            if entry is None:
                raise RuntimeError("a query leaf has no in-bag training observations")
            indices, tree_weights = entry
            weights[indices] += tree_weights
        weights /= self.n_trees
        return np.flatnonzero(weights > 0), weights

    def apply(self, X):
        leaves = np.asarray(self.forest.apply(X))
        if leaves.shape != (len(X), self.n_trees):
            raise ValueError("forest.apply(X) returned an invalid shape")
        return leaves

