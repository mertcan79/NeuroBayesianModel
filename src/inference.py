import numpy as np
import pandas as pd
from typing import Dict, Any, List
from bayesian_node import BayesianNode, CategoricalNode, Node
import logging
from scipy.stats import rv_continuous, rv_discrete


logger = logging.getLogger()
logger.setLevel(logging.WARNING) 

class Inference:
    def __init__(self, nodes):
        self.nodes = nodes
        self.sampled_values = {}

    def __repr__(self):
        return f"Inference(nodes={list(self.nodes.keys())})"

    def sample_node(self, node_name: str, size: int = 1) -> np.ndarray:
        node = self.nodes[node_name]
        if isinstance(node, CategoricalNode):
            return self._sample_categorical_node(node, size)
        else:
            return self._sample_continuous_node(node, size)

    def _sample_categorical_node(self, node: CategoricalNode, size: int) -> np.ndarray:
        if not node.fitted:
            raise ValueError(f"Node '{node.name}' has not been fitted yet.")
        
        if node.parents:
            parent_samples = [self.sample_node(parent.name, size) for parent in node.parents]
            parent_values = np.column_stack(parent_samples)
            samples = np.array([node.sample(1, tuple(pv))[0] for pv in parent_values])
        else:
            probs = np.array(list(node.distribution.values()))
            samples = np.random.choice(node.categories, size=size, p=probs)
        
        return samples

    def _sample_continuous_node(self, node: BayesianNode, size: int) -> np.ndarray:
        if not node.fitted:
            raise ValueError(f"Node '{node.name}' has not been fitted yet.")
        
        if node.parents:
            parent_samples = [self.sample_node(parent.name, size) for parent in node.parents]
            parent_values = np.column_stack(parent_samples)
            mean = node.distribution['intercept'] + parent_values @ node.distribution['beta']
            samples = np.random.normal(mean, node.distribution['std'], size=size)
        else:
            samples = np.random.normal(node.distribution['mean'], node.distribution['std'], size=size)
        
        return samples
            
    def log_likelihood(self, nodes, data):
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

    def infer_with_missing_data(self, data: pd.DataFrame, num_samples: int = 1000) -> pd.DataFrame:
        results = pd.DataFrame()
        for _, row in data.iterrows():
            observed = {col: val for col, val in row.items() if pd.notna(val)}
            sampled = self.gibbs_sampling(observed, num_samples=num_samples)
            results = results.append(sampled, ignore_index=True)
        return results

    def gibbs_sampling(self, observed: Dict[str, Any], num_samples: int) -> pd.DataFrame:
        samples = {node: [] for node in self.nodes}
        current_sample = {node: np.random.choice(self.nodes[node].categories) if isinstance(self.nodes[node], CategoricalNode)
                          else np.random.normal(size=1)[0] for node in self.nodes}
        
        for i in range(num_samples):
            for node in self.nodes:
                if node not in observed:
                    parents = self.get_parents(node)
                    if not parents:
                        if isinstance(self.nodes[node], CategoricalNode):
                            probs = self.nodes[node].probabilities
                            current_sample[node] = np.random.choice(self.nodes[node].categories, p=probs)
                        else:
                            current_sample[node] = self.sample_node(node, 1)[0]
                    else:
                        parent_values = [current_sample[p] for p in parents]
                        if isinstance(self.nodes[node], CategoricalNode):
                            probs = self.nodes[node].conditional_probabilities(tuple(parent_values))
                            current_sample[node] = np.random.choice(self.nodes[node].categories, p=probs)
                        else:
                            beta = self.nodes[node].params.get('beta', np.zeros(len(parents)))
                            loc = np.dot(parent_values, beta)
                            scale = self.nodes[node].params.get('scale', 1.0)
                            current_sample[node] = np.random.normal(loc, scale)
                    
                    samples[node].append(current_sample[node])
        
        return pd.DataFrame(samples)
