import numpy as np
import pandas as pd
from typing import Dict
from scipy import stats
import logging
from scipy.stats import norm, gamma, dirichlet
import statsmodels.api as sm
from bayesian_node import BayesianNode, CategoricalNode

logger = logging.getLogger(__name__)

def fit_parameters(nodes, data):
    for node_name, node in nodes.items():
        node_data = data[node_name]
        parent_data = data[[parent.name for parent in node.parents]] if node.parents else None
        node.fit(node_data, parent_data)

def fit_continuous_parameters(node, node_data, parent_data):
    if parent_data is None:
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

        node.params = {
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

        node.params = {
            'mean': posterior_mean,
            'precision': posterior_precision,
            'df': posterior_df,
            'scale': posterior_scale,
            'beta': beta
        }

def fit_categorical_parameters(node, node_data, parent_data):
    if parent_data is None:
        # Dirichlet conjugate prior
        prior_counts = np.ones(len(node.categories))
        counts = np.bincount(node_data, minlength=len(node.categories))
        posterior_counts = prior_counts + counts
        node.params = {'dirichlet_params': posterior_counts}
    else:
        # Dirichlet for each parent combination
        parent_combinations = parent_data.apply(tuple, axis=1).unique()
        node.params = {}
        for combination in parent_combinations:
            mask = (parent_data.apply(tuple, axis=1) == combination)
            counts = np.bincount(node_data[mask], minlength=len(node.categories))
            prior_counts = np.ones(len(node.categories))
            posterior_counts = prior_counts + counts
            node.params[combination] = {'dirichlet_params': posterior_counts}