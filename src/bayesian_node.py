from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import List
from scipy.stats import multinomial
from scipy import stats

class BayesianNode:
    def __init__(self, name: str):
        self.name = name
        self.parents = []
        self.children = []
        self.distribution = None
        self.params = {}
        self.scaler = StandardScaler()

    def fit_scaler(self, data):
        if np.issubdtype(data.dtype, np.number):
            self.scaler.fit(data.reshape(-1, 1))
        else:
            # For non-numeric data, don't use a scaler
            pass

    def transform(self, data):
        if np.issubdtype(data.dtype, np.number):
            return self.scaler.transform(data.reshape(-1, 1)).flatten()
        else:
            return data

    def inverse_transform(self, data):
        if np.issubdtype(data.dtype, np.number):
            return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        else:
            return data

    def set_distribution(self, distribution, params=None):
        self.distribution = distribution
        self.params = params or {}

class CategoricalNode(BayesianNode):
    def __init__(self, name: str, categories: List[str]):
        super().__init__(name)
        self.categories = categories
        self.category_map = {cat: i for i, cat in enumerate(categories)}

    def fit_scaler(self, data):
        # No scaling for categorical data
        pass

    def transform(self, data):
        return np.array([self.category_map.get(d, -1) for d in data])

    def inverse_transform(self, data):
        inv_map = {v: k for k, v in self.category_map.items()}
        return np.array([inv_map.get(d, 'Unknown') for d in data])

    def set_distribution(self, counts):
        super().set_distribution(stats.multinomial, params={'n': 1, 'p': counts / np.sum(counts)})

    def sample(self, size=1):
        return self.inverse_transform(self.distribution.rvs(size=size, **self.params))