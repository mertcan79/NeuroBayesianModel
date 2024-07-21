from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import List
from scipy.stats import multinomial
from scipy import stats
import pandas as pd

class BayesianNode:
    def __init__(self, name: str):
        self.name = name
        self.parents = []
        self.children = []
        self.distribution = None
        self.params = {}
        self.scaler = StandardScaler()
        self.fitted = False  # Add a flag to check if the scaler is fitted

    def fit_scaler(self, data):
        if np.issubdtype(data.dtype, np.number):
            self.scaler.fit(data.reshape(-1, 1))
            self.fitted = True

    def transform(self, data):
        if np.issubdtype(data.dtype, np.number):
            if not self.fitted:
                raise ValueError("Scaler is not fitted yet.")
            return self.scaler.transform(data.reshape(-1, 1)).flatten()
        else:
            return data

    def inverse_transform(self, data):
        if np.issubdtype(data.dtype, np.number):
            if not self.fitted:
                raise ValueError("Scaler is not fitted yet.")
            return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        else:
            return data

    def set_distribution(self, distribution, params=None):
        self.distribution = distribution
        self.params = params or {}

    def sample(self, size: int = 1, parent_samples=None) -> np.ndarray:
        if not self.parents:
            if self.distribution is not None:
                samples = self.distribution.rvs(size=size, **self.params)
                return self.inverse_transform(samples)
            else:
                raise ValueError(f"No distribution set for node {self.name}")
        else:
            if parent_samples is None:
                raise ValueError("Parent samples must be provided for nodes with parents")
            parent_values = np.column_stack([parent_samples[parent.name] for parent in self.parents])
            beta = self.params.get('beta', np.zeros(len(self.parents)))
            intercept = self.params.get('intercept', 0)
            scale = self.params.get('scale', 1.0)
            loc = intercept + np.dot(parent_values, beta)
            noise = np.random.normal(0, scale, size)
            samples = loc + noise
            return self.inverse_transform(samples)

class CategoricalNode(BayesianNode):
    def __init__(self, name: str, categories: List[str]):
        super().__init__(name)
        self.categories = categories
        self.category_map = {cat: i for i, cat in enumerate(categories)}

    def transform(self, data):
        return np.array([self.category_map.get(d, -1) for d in data])

    def inverse_transform(self, data):
        inv_map = {v: k for k, v in self.category_map.items()}
        return np.array([inv_map.get(d, 'Unknown') for d in data])

    def set_distribution(self, counts):
        super().set_distribution(stats.multinomial, params={'n': 1, 'p': counts / np.sum(counts)})

    def sample(self, size: int = 1, parent_samples=None) -> np.ndarray:
        if self.distribution is not None:
            samples = self.distribution.rvs(size=size, **self.params)
            return self.inverse_transform(samples)
        else:
            raise ValueError(f"No distribution set for node {self.name}")