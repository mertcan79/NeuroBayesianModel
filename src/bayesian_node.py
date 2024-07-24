import numpy as np
from typing import List, Tuple, Union
from scipy import stats
import pandas as pd

class BayesianNode:
    def __init__(self, name: str):
        self.name = name
        self.parents: List['BayesianNode'] = []
        self.children: List['BayesianNode'] = []
        self.parameters = None
        self.distribution = None
        self.is_categorical = False
        self.categories = None

    def __eq__(self, other):
        if isinstance(other, BayesianNode):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)

    def add_parent(self, parent: 'BayesianNode'):
        if parent not in self.parents:
            self.parents.append(parent)

    def add_child(self, child: 'BayesianNode'):
        if child not in self.children:
            self.children.append(child)

    def set_parameters(self, parameters):
        self.parameters = parameters

    def set_distribution(self, distribution):
        self.distribution = distribution

    def set_categorical(self, is_categorical: bool, categories: List = None):
        self.is_categorical = is_categorical
        self.categories = categories

    def fit(self, node_data, parent_data=None):
        if self.is_categorical:
            self.fit_categorical(node_data, parent_data)
        else:
            self.fit_continuous(node_data, parent_data)

    def fit_categorical(self, node_data, parent_data=None):
        if parent_data is None or parent_data.empty:
            value_counts = node_data.value_counts(normalize=True)
            self.distribution = value_counts.to_dict()
        else:
            joint_counts = pd.crosstab(parent_data.apply(tuple, axis=1), node_data)
            self.distribution = (joint_counts / joint_counts.sum(axis=1)).to_dict()

    def fit_continuous(self, node_data, parent_data):
        if parent_data is not None and len(parent_data.columns) > 0:
            X = parent_data.values
            y = node_data.values
            print(f"X shape: {X.shape}, y shape: {y.shape}")
            print(f"X columns: {parent_data.columns}")
            print(f"y name: {node_data.name}")
            try:
                X = np.column_stack((np.ones(X.shape[0]), X))
                model, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
                self.distribution = (model, np.std(residuals))
            except Exception as e:
                print(f"Error in regression: {str(e)}")
                raise
        else:
            self.distribution = (np.mean(node_data), np.std(node_data))

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def sample(self, size: int = 1, parent_values: dict = None) -> np.ndarray:
        if self.distribution is None:
            raise ValueError(f"Distribution for node {self.name} is not set")
        
        if self.is_categorical:
            probs = list(self.distribution.values())
            return np.random.choice(list(self.distribution.keys()), size=size, p=probs)
        else:
            mean, std = self.distribution
            if parent_values:
                parent_array = np.array([1] + [parent_values[p.name] for p in self.parents])
                mean = np.dot(mean, parent_array)
            return np.random.normal(mean, std, size=size)

    def log_probability(self, value: Union[float, str], parent_values: Tuple = None) -> float:
        if self.distribution is None:
            raise ValueError(f"Distribution for node {self.name} is not set")
        
        if self.is_categorical:
            if value not in self.distribution:
                return float('-inf')
            return np.log(self.distribution[value])
        else:
            mean, std = self.distribution
            if parent_values:
                parent_array = np.array([1] + list(parent_values))
                mean = np.dot(mean, parent_array)
            return stats.norm.logpdf(value, mean, std)

    def __repr__(self):
        return f"BayesianNode(name={self.name}, parents={[p.name for p in self.parents]}, children={[c.name for c in self.children]})"

class CategoricalNode(BayesianNode):
    def __init__(self, name, categories):
        super().__init__(name)
        self.categories = list(range(len(categories)))  # Use integer codes
        self.original_categories = categories
        self.distribution = stats.multinomial
        self.cpt = None

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'categories': self.categories,
            'original_categories': self.original_categories,
            'cpt': self.cpt.tolist() if self.cpt is not None else None
        })
        return base_dict

    @classmethod
    def from_dict(cls, data):
        node = cls(data['name'], data['original_categories'])
        node.params = data['params']
        node.categories = data['categories']
        node.cpt = np.array(data['cpt']) if data['cpt'] is not None else None
        return node

    def fit(self, data, parent_data=None):
        if parent_data is None or len(parent_data) == 0:
            counts = np.bincount(data, minlength=len(self.categories))
            self.distribution = stats.multinomial(n=1, p=counts / np.sum(counts))
            self.params = {"p": self.distribution.p}
            self.cpt = self.params["p"]
        else:
            parent_combinations = np.array(np.meshgrid(*[range(len(set(parent_data[col]))) for col in parent_data.columns])).T.reshape(-1, parent_data.shape[1])
            
            self.cpt = np.zeros((len(parent_combinations), len(self.categories)))
            for i, parent_comb in enumerate(parent_combinations):
                mask = np.all(parent_data == parent_comb, axis=1)
                counts = np.bincount(data[mask], minlength=len(self.categories))
                self.cpt[i] = counts / np.sum(counts)

            self.params = {"cpt": self.cpt}
        return self  # Return self to allow method chaining

    def log_probability(self, value, parent_values=None):
        if self.cpt is None:
            raise ValueError("Distribution not fitted yet")

        if parent_values is None or len(parent_values) == 0:
            return np.log(self.cpt[value])
        else:
            parent_index = np.ravel_multi_index(parent_values, [len(set(parent_values[i])) for i in range(len(parent_values))])
            return np.log(self.cpt[parent_index, value])

    def sample(self, size=1, parent_samples=None):
        probs = self.get_conditional_probabilities(parent_samples)
        return np.random.choice(self.categories, size=size, p=probs)

    def get_conditional_probabilities(self, parent_samples=None):
        if parent_samples is None or len(parent_samples) == 0:
            return self.cpt
        else:
            parent_index = np.ravel_multi_index(parent_samples, [len(set(parent_samples[i])) for i in range(len(parent_samples))])
            return self.cpt[parent_index]