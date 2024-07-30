import numpy as np
from typing import List, Tuple, Union, Optional, Dict
from scipy import stats
import pandas as pd
import statsmodels.api as sm

class BayesianNode:
    def __init__(self, name: str, distribution=None):
        self.name = name
        self.parents = []
        self.params = None
        self.distribution = distribution
        self.is_categorical = False
        self.categories = None
        self.transform = None
        self.inverse_transform = None
        self.fitted = False
        self.children = []
        self.distribution_type = None
        self.coefficients = None
        self.intercept = None
        self.std = None
        self.mean = None

    def __repr__(self):
        return f"BayesianNode(name={self.name}, distribution={self.distribution}, parents={self.parents})"

    def __eq__(self, other):
        if isinstance(other, BayesianNode):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)

    def to_dict(self):
        return {
            'name': self.name,
            'parents': [parent.name for parent in self.parents],
            'children': [child.name for child in self.children],
            'params': self.params,
            'distribution': str(self.distribution) if self.distribution else None
        }

    @classmethod
    def from_dict(cls, data):
        node = cls(data['name'])
        node.params = data['params']
        node.distribution = eval(data['distribution']) if data['distribution'] else None
        return node

    def add_parent(self, parent: 'BayesianNode'):
        if parent not in self.parents:
            self.parents.append(parent)

    def add_child(self, child: 'BayesianNode'):
        if child not in self.children:
            self.children.append(child)

    def set_params(self, params):
        self.params = params

    def set_distribution(self, distribution):
        self.distribution = distribution
        if isinstance(distribution, dict):
            if 'mean' in distribution and 'std' in distribution:
                self.distribution_type = 'gaussian'
            elif 'beta' in distribution:
                self.distribution_type = 'linear'
        elif isinstance(distribution, (stats.rv_continuous, stats.rv_discrete)):
            self.distribution_type = 'scipy'
        else:
            raise ValueError(f"Unsupported distribution type for node {self.name}")

    def get_distribution(self):
        if isinstance(self.distribution, (stats.rv_continuous, stats.rv_discrete)):
            return self.distribution
        elif isinstance(self.distribution, tuple) and len(self.distribution) == 2:
            return stats.norm(*self.distribution)
        else:
            return self.distribution

    def set_categorical(self, categories):
        self.is_categorical = True
        self.categories = list(categories)

    def set_transform(self, transform, inverse_transform):
        self.transform = transform
        self.inverse_transform = inverse_transform

    def apply_transform(self, data):
        if self.transform:
            return self.transform(data)
        return data

    def apply_inverse_transform(self, data):
        if self.inverse_transform:
            return self.inverse_transform(data)
        return data

    def get_conditional_mean(self, parent_values):
        if self.parents:
            return self.distribution['intercept'] + np.dot(parent_values, self.distribution['beta'])
        else:
            return self.distribution['mean']

    def sample(self, size=1, parent_values=None):
        if self.distribution_type == 'gaussian':
            return np.random.normal(self.distribution['mean'], self.distribution['std'], size)
        elif self.distribution_type == 'linear':
            if parent_values is None:
                raise ValueError("Parent values required for linear distribution")
            return self.distribution['intercept'] + np.dot(parent_values, self.distribution['beta']) + \
                   np.random.normal(0, self.distribution['std'], size)
        elif self.distribution_type == 'scipy':
            return self.distribution.rvs(size=size)
        else:
            raise ValueError(f"Unsupported distribution type for node {self.name}")

    def fit(self, node_data, parent_data=None):
        if self.is_categorical or pd.api.types.is_categorical_dtype(node_data):
            self.fit_categorical(node_data, parent_data)
        else:
            self.fit_continuous(node_data, parent_data)

    def fit_categorical(self, node_data, parent_data):
        self.is_categorical = True
        self.categories = sorted(node_data.unique())
        
        if parent_data is None or parent_data.empty:
            counts = node_data.value_counts(normalize=True)
            self.distribution = counts.reindex(self.categories, fill_value=0).values
        else:
            self.distribution = {}
            for parent_combo in parent_data.drop_duplicates().itertuples(index=False):
                mask = (parent_data == parent_combo).all(axis=1)
                counts = node_data[mask].value_counts(normalize=True)
                self.distribution[parent_combo] = counts.reindex(self.categories, fill_value=0).values

        self.fitted = True

    def fit_continuous(self, node_data, parent_data):
        if parent_data is None or parent_data.empty:
            self.distribution = {
                'mean': np.mean(node_data),
                'std': np.std(node_data) if len(node_data) > 1 else 1e-6
            }
        else:
            X = sm.add_constant(parent_data)
            model = sm.OLS(node_data, X).fit()
            self.distribution = {
                'intercept': model.params.iloc[0] if not model.params.empty else None,
                'coefficients': model.params[1:].values if len(model.params) > 1 else None,
                'std': model.resid.std() if not model.resid.empty else 1e-6
            }

        self.fitted = True


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

    def compute_sensitivity(self, target_node: str, evidence_nodes: List[str]) -> Dict[str, float]:
        if self.inference is None:
            raise ValueError("Inference object not initialized. Call fit() first.")
        return self.inference.compute_sensitivity(target_node, evidence_nodes)

class CategoricalNode(BayesianNode):
    def __init__(self, name, categories=None):
        super().__init__(name)
        self.categories = list(categories) if categories is not None else []
        self.original_categories = categories
        self.distribution = None
        self.is_categorical = True

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
        node = cls(data['name'], data['categories'])
        node.cpt = np.array(data['cpt']) if data['cpt'] is not None else None
        return node

    def fit(self, data, parent_data=None):
        if parent_data is not None:
            self._fit_with_parents(data, parent_data)
        else:
            counts = data.value_counts()
            self.distribution = (counts / counts.sum()).to_dict()
            self.fitted = True

    def set_distribution(self, dist):
        self.distribution = dist

    def get_distribution(self):
        return self.distribution

    def add_parent(self, parent: 'CategoricalNode'):
        if parent not in self.parents:
            self.parents.append(parent)

    def add_child(self, child: 'CategoricalNode'):
        if child not in self.children:
            self.children.append(child)

    def set_categorical(self, categories):
        self.categories = list(categories)
        self.original_categories = categories

    def _fit_with_parents(self, data, parent_data):
        unique_parent_combinations = parent_data.drop_duplicates()
        self.distribution = {}
        for _, parent_combination in unique_parent_combinations.iterrows():
            parent_values = tuple(parent_combination)
            subset = data[parent_data.isin(parent_values).all(axis=1)]
            counts = subset.value_counts()
            self.distribution[parent_values] = (counts / counts.sum()).to_dict()
        self.fitted = True

    def log_probability(self, values, parent_values=None):
        if parent_values is not None:
            return self._log_probability_with_parents(values, parent_values)
        else:
            prob = [self.distribution.get(value, 1e-10) for value in values]
            return np.log(np.prod(prob))

    def _log_probability_with_parents(self, values, parent_values):
        prob = 1.0
        for value, parent_value in zip(values, parent_values):
            distribution = self.distribution.get(parent_value, {})
            prob *= distribution.get(value, 1e-10)
        return np.log(prob)

    def sample(self, size=1, parent_values=None):
        if parent_values is not None:
            return self._sample_with_parents(size, parent_values)
        else:
            if not self.fitted:
                raise ValueError("Node has not been fitted yet.")
            choices = np.random.choice(self.categories, size=size, p=list(self.distribution.values()))
            return choices

    def _sample_with_parents(self, size, parent_values):
        if not self.fitted:
            raise ValueError("Node has not been fitted yet.")
        distribution = self.distribution.get(parent_values, {})
        if not distribution:
            raise ValueError(f"No distribution for parent values {parent_values}.")
        choices = np.random.choice(self.categories, size=size, p=list(distribution.values()))
        return choices

    def get_conditional_probs(self, parent_values):
        if not self.parents:
            return self.distribution
        
        parent_values = tuple(parent_values)
        if parent_values in self.distribution:
            return self.distribution[parent_values]
        else:
            return np.ones(len(self.categories)) / len(self.categories)

    def get_conditional_mean(self, parent_values):
        probs = self.get_conditional_probs(parent_values)
        return np.dot(range(len(self.categories)), probs)

Node = Union[BayesianNode, CategoricalNode]
