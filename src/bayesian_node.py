import numpy as np
from typing import List, Tuple, Union
from scipy import stats
import pandas as pd
from scipy.stats import norm, gamma, dirichlet

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

    def to_dict(self):
        return {
            'name': self.name,
            'parents': [parent.name for parent in self.parents],
            'children': [child.name for child in self.children],
            'parameters': self.parameters,
            'distribution': str(self.distribution) if self.distribution else None
        }

    @classmethod
    def from_dict(cls, data):
        node = cls(data['name'])
        node.parameters = data['parameters']
        node.distribution = eval(data['distribution']) if data['distribution'] else None
        return node

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

    def fit_continuous(self, node_data, parent_data):
        if parent_data is None or parent_data.empty:
            # Normal-Gamma conjugate prior
            prior_mean = 0
            prior_precision = 1
            prior_shape = 1
            prior_rate = 1

            n = len(node_data)
            sample_mean = np.mean(node_data)
            sample_var = np.var(node_data)

            posterior_mean = (prior_precision * prior_mean + n * sample_mean) / (prior_precision + n)
            posterior_precision = prior_precision + n
            posterior_shape = prior_shape + n / 2
            posterior_rate = prior_rate + 0.5 * (n * sample_var + prior_precision * n * (sample_mean - prior_mean)**2 / (prior_precision + n))

            self.distribution = {
                'mean': posterior_mean,
                'precision': posterior_precision,
                'shape': posterior_shape,
                'rate': posterior_rate
            }
        else:
            # Multivariate Normal-Wishart for multiple parents
            X = parent_data.values
            y = node_data.values

            n, d = X.shape
            prior_mean = np.zeros(d)
            prior_precision = np.eye(d)
            prior_df = d + 2
            prior_scale = np.eye(d)

            X_mean = X.mean(axis=0)
            S = np.cov(X, rowvar=False) * (n - 1)
            beta = np.linalg.solve(S, np.dot(X.T, y))

            posterior_mean = np.linalg.solve(prior_precision + n * np.linalg.inv(S), 
                                             np.dot(prior_precision, prior_mean) + n * np.dot(np.linalg.inv(S), X_mean))
            posterior_precision = prior_precision + n * np.linalg.inv(S)
            posterior_df = prior_df + n
            posterior_scale = prior_scale + S + \
                              n * np.outer(X_mean - posterior_mean, np.dot(np.linalg.inv(S), X_mean - posterior_mean))

            self.distribution = {
                'mean': posterior_mean,
                'precision': posterior_precision,
                'df': posterior_df,
                'scale': posterior_scale,
                'beta': beta
            }

    def fit_categorical(self, node_data, parent_data):
        if parent_data is None or parent_data.empty:
            # Dirichlet conjugate prior
            prior_counts = np.ones(len(self.categories))
            counts = np.bincount(node_data, minlength=len(self.categories))
            posterior_counts = prior_counts + counts
            self.distribution = dirichlet(posterior_counts)
        else:
            # Dirichlet for each parent combination
            parent_combinations = parent_data.apply(tuple, axis=1).unique()
            self.distribution = {}
            for combination in parent_combinations:
                mask = (parent_data.apply(tuple, axis=1) == combination)
                counts = np.bincount(node_data[mask], minlength=len(self.categories))
                prior_counts = np.ones(len(self.categories))
                posterior_counts = prior_counts + counts
                self.distribution[combination] = dirichlet(posterior_counts)

    def sample(self, size: int, parent_values: pd.DataFrame = None) -> np.ndarray:
        if parent_values is not None and not parent_values.empty:
            if isinstance(self.distribution, dict):
                # Handle categorical nodes with parents
                parent_combinations = tuple(parent_values.apply(tuple, axis=1))
                if parent_combinations in self.distribution:
                    probs = self.distribution[parent_combinations].rvs(size=size)
                    return probs
                else:
                    raise ValueError(f"Parent combination not found in distribution for node {self.name}")
            elif isinstance(self.distribution, tuple):
                if len(self.distribution) == 2:  # Gaussian distribution
                    mean, std = self.distribution
                    return np.random.normal(mean, std, size=size)
                elif len(self.distribution) > 2:  # Linear model
                    intercept, *coefficients = self.distribution
                    parent_values_array = parent_values.values
                    mean = intercept + np.sum(parent_values_array * coefficients, axis=1)
                    return np.random.normal(mean, scale=1.0, size=size)  # Assuming unit variance
            elif isinstance(self.distribution, stats.rv_continuous):
                return self.distribution.rvs(size=size)
            elif isinstance(self.distribution, stats.rv_discrete):
                return self.distribution.rvs(size=size)
            elif callable(self.distribution):
                return self.distribution(parent_values, size)
            else:
                raise ValueError(f"Unsupported distribution type for node {self.name}")
        else:
            if isinstance(self.distribution, tuple):
                if len(self.distribution) == 2:  # Gaussian distribution
                    mean, std = self.distribution
                    std = abs(std)  # Ensure std is non-negative
                    return np.random.normal(mean, std, size=size)
            elif isinstance(self.distribution, stats.rv_continuous):
                return self.distribution.rvs(size=size)
            elif isinstance(self.distribution, stats.rv_discrete):
                return self.distribution.rvs(size=size)
            elif callable(self.distribution):
                return self.distribution(size=size)
            else:
                raise ValueError(f"Unsupported distribution type for node {self.name}")

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
    def __init__(self, name, categories, parameters):
        super().__init__(name)
        self.name = name
        self.categories = list(range(len(categories)))  # Use integer codes
        self.original_categories = categories
        self.distribution = stats.multinomial
        self.cpt = None
        self.parameters = parameters

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
        node = cls(data['name'], data['categories'], data['parameters'])
        node.cpt = np.array(data['cpt']) if data['cpt'] is not None else None
        return node

    def fit(self, data, parent_data=None):
        if parent_data is None or len(parent_data) == 0:
            counts = np.bincount(data, minlength=len(self.categories))
            self.distribution = stats.multinomial(n=1, p=counts / np.sum(counts))
            self.parameters = {"p": self.distribution.p}
            self.cpt = self.parameters["p"]
        else:
            parent_combinations = np.array(np.meshgrid(*[range(len(set(parent_data[col]))) for col in parent_data.columns])).T.reshape(-1, parent_data.shape[1])
            
            self.cpt = np.zeros((len(parent_combinations), len(self.categories)))
            for i, parent_comb in enumerate(parent_combinations):
                mask = np.all(parent_data == parent_comb, axis=1)
                counts = np.bincount(data[mask], minlength=len(self.categories))
                self.cpt[i] = counts / np.sum(counts)

            self.parameters = {"cpt": self.cpt}
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