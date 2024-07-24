import numpy as np
import pandas as pd
from typing import Dict
from scipy import stats
import logging
from scipy.stats import norm, gamma, dirichlet

from .bayesian_node import BayesianNode, CategoricalNode

logger = logging.getLogger(__name__)


def fit_parameters(nodes, data):
    for node_name, node in nodes.items():
        node_data = data[node_name]
        parent_data = data[[parent.name for parent in node.parents]] if node.parents else None

        if isinstance(node, CategoricalNode):
            node.fit(node_data, parent_data)
        else:
            # For continuous nodes, we'll use a simple linear regression
            if parent_data is not None:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression().fit(parent_data, node_data)
                node.params = {
                    'intercept': model.intercept_,
                    'coefficients': model.coef_
                }
            else:
                node.params = {
                    'mean': node_data.mean(),
                    'std': node_data.std()
                }
        
        node.fitted = True

