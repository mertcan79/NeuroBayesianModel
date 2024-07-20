import numpy as np
import pandas as pd
from typing import Dict
from bayesian_node import BayesianNode
import logging

# Set up logging configuration
logger = logging.getLogger()
logger.setLevel(logging.WARNING)  # Set to WARNING to suppress DEBUG and INFO logs

def log_likelihood(nodes: Dict[str, BayesianNode], data: pd.DataFrame) -> float:
    log_likelihood = 0.0
    for _, row in data.iterrows():
        for node_name, node in nodes.items():
            scaled_value = node.transform(row[node_name])
            if not node.parents:
                log_likelihood += node.distribution.logpdf(scaled_value, **node.params)
            else:
                parent_values = np.array([nodes[p.name].transform(row[p.name]) for p in node.parents])
                if parent_values.ndim == 1:
                    parent_values = parent_values.reshape(-1, 1)
                loc = np.dot(parent_values.T, node.params['beta'])
                scale = node.params['scale']
                log_likelihood += node.distribution.logpdf(scaled_value, loc=loc, scale=scale)
    return float(log_likelihood)

def sample_node(nodes: Dict[str, BayesianNode], node_name: str, size: int = 1) -> np.ndarray:
    node = nodes[node_name]
    if not node.parents:
        samples = node.distribution.rvs(size=size, **node.params)
    else:
        parent_values = np.column_stack([sample_node(nodes, p.name, size) for p in node.parents])
        parent_values_scaled = np.column_stack([nodes[p.name].transform(parent_values[:, i]) for i, p in enumerate(node.parents)])
        loc = np.dot(parent_values_scaled, node.params['beta'])
        samples = node.distribution.rvs(loc=loc, scale=node.params['scale'], size=size)
    
    # Log summary of samples instead of detailed values
    logger.debug(f"Sampled values for node {node_name}: Mean = {np.mean(samples):.4f}, Std = {np.std(samples):.4f}")
    
    return node.inverse_transform(samples.reshape(-1, 1))

def infer_with_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame()
    for _, row in data.iterrows():
        observed = {col: val for col, val in row.items() if pd.notna(val)}
        sampled = self.metropolis_hastings(observed, num_samples=1000)
        results = results.append(sampled, ignore_index=True)
    return results