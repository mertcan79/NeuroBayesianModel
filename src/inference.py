import numpy as np
import pandas as pd
from typing import Dict
from bayesian_node import BayesianNode, CategoricalNode
import logging

# Set up logging configuration
logger = logging.getLogger()
logger.setLevel(logging.WARNING)  # Set to WARNING to suppress DEBUG and INFO logs

def log_likelihood(nodes, data):
    log_likelihood = 0.0
    for _, row in data.iterrows():
        for node_name, node in nodes.items():
            scaled_value = node.transform(row[node_name])
            if node.distribution is not None:
                log_likelihood += node.distribution.logpdf(scaled_value, **node.params)
            else:
                logger.warning(f"No distribution set for node {node_name}")
    return log_likelihood

def sample_node(self, node_name: str, size: int = 1) -> np.ndarray:
    node = self.nodes[node_name]
    if not node.parents:
        if isinstance(node, CategoricalNode):
            return node.sample(size)
        elif node.distribution is not None:
            samples = node.distribution.rvs(size=size, **node.params)
            return node.inverse_transform(samples)
        else:
            raise ValueError(f"No distribution set for node {node_name}")
    else:
        parent_values = np.column_stack([self.sample_node(p.name, size) for p in node.parents])
        if isinstance(node, CategoricalNode):
            return node.sample(size)
        else:
            beta = node.params.get('beta', np.zeros(len(node.parents)))
            loc = np.dot(parent_values, beta)
            scale = node.params.get('scale', 1.0)
            noise = np.random.normal(0, scale, size)
            samples = loc + noise
            return node.inverse_transform(samples)


def infer_with_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame()
    for _, row in data.iterrows():
        observed = {col: val for col, val in row.items() if pd.notna(val)}
        sampled = self.metropolis_hastings(observed, num_samples=1000)
        results = results.append(sampled, ignore_index=True)
    return results