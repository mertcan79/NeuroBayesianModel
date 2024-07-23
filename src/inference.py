import numpy as np
import pandas as pd
from typing import Dict
from .bayesian_node import BayesianNode, CategoricalNode
import logging

# Set up logging configuration
logger = logging.getLogger()
logger.setLevel(logging.WARNING)  # Set to WARNING to suppress DEBUG and INFO logs


def log_likelihood(nodes, data):
    total_ll = 0
    if isinstance(data, pd.DataFrame):
        for _, row in data.iterrows():
            row_ll = 0
            for node_name, node in nodes.items():
                parents = node.parents
                if parents:
                    parent_values = [row[parent.name] for parent in parents]
                    row_ll += node.log_probability(row[node_name], parent_values)
                else:
                    row_ll += node.log_probability(row[node_name])
            total_ll += row_ll
    else:  # Assume it's a single row (dictionary or Series)
        for node_name, node in nodes.items():
            parents = node.parents
            if parents:
                parent_values = [data[parent.name] for parent in parents]
                total_ll += node.log_probability(data[node_name], parent_values)
            else:
                total_ll += node.log_probability(data[node_name])
    return total_ll


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
        parent_values = np.column_stack(
            [self.sample_node(p.name, size) for p in node.parents]
        )
        if isinstance(node, CategoricalNode):
            return node.sample(size)
        else:
            beta = node.params.get("beta", np.zeros(len(node.parents)))
            loc = np.dot(parent_values, beta)
            scale = node.params.get("scale", 1.0)
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
