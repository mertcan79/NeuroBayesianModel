import numpy as np
from scipy import stats
from typing import List, Union, Callable
import pymc as pm
import arviz as az

class BayesianNode:
    def __init__(self, name: str, distribution: Union[stats.rv_continuous, stats.rv_discrete, Callable], 
                 children: List['BayesianNode'] = None, parents: List['BayesianNode'] = None):
        self.name = name
        self.distribution = distribution
        self.children = children or []
        self.parents = parents or []
        self.data = []

    def update(self, data: Union[float, List[float]]):
        if not isinstance(data, list):
            data = [data]
        self.data.extend(data)
        self._update_distribution()

    def _update_distribution(self):
        if callable(self.distribution):
            # For conditional distributions, we'll update in the BayesianNetwork class
            pass
        elif isinstance(self.distribution, stats.rv_continuous):
            with pm.Model() as model:
                mu = pm.Normal("mu", mu=np.mean(self.data), sigma=np.std(self.data))
                sigma = pm.HalfNormal("sigma", sigma=np.std(self.data))
                likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=self.data)
                trace = pm.sample(2000, tune=1000, return_inferencedata=True)
            
            posterior_mu = az.summary(trace)["mean"]["mu"].values[0]
            posterior_sigma = az.summary(trace)["mean"]["sigma"].values[0]
            self.distribution = stats.norm(loc=posterior_mu, scale=posterior_sigma)
        
        elif isinstance(self.distribution, stats.rv_discrete):
            counts = np.bincount(self.data)
            self.distribution = stats.rv_discrete(values=(np.arange(len(counts)), counts / len(self.data)))

    def sample(self, size: int = 1, parent_values: np.ndarray = None) -> np.ndarray:
        if callable(self.distribution):
            if parent_values is None:
                raise ValueError("Parent values must be provided for conditional sampling")
            return self.distribution(parent_values).rvs(size=size)
        else:
            return self.distribution.rvs(size=size)

    def probability(self, value: Union[float, np.ndarray], parent_values: np.ndarray = None) -> np.ndarray:
        if callable(self.distribution):
            if parent_values is None:
                raise ValueError("Parent values must be provided for conditional probability")
            return self.distribution(parent_values).pdf(value)
        elif isinstance(self.distribution, (stats.rv_continuous, stats._distn_infrastructure.rv_frozen)):
            return self.distribution.pdf(value)
        elif isinstance(self.distribution, stats.rv_discrete):
            return self.distribution.pmf(value)
        else:
            raise ValueError(f"Unsupported distribution type: {type(self.distribution)}")

    def credible_interval(self, alpha: float = 0.05) -> tuple:
        return self.distribution.interval(1 - alpha)

    def copy(self) -> 'BayesianNode':
        new_dist = self.distribution
        if isinstance(self.distribution, stats._distn_infrastructure.rv_frozen):
            dist_type = self.distribution.dist
            new_dist = dist_type(**self.distribution.kwds)
        return BayesianNode(self.name, new_dist, children=self.children.copy(), parents=self.parents.copy())