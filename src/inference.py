import numpy as np
import pandas as pd
from typing import Dict, Any, List
from bayesian_node import BayesianNode, CategoricalNode, Node
import logging

# Set up logging configuration
logger = logging.getLogger()
logger.setLevel(logging.WARNING)  # Set to WARNING to suppress DEBUG and INFO logs

class Inference:
    def __init__(self, nodes: Dict[str, BayesianNode]):
        self.nodes = nodes
    
    def set_nodes(self, nodes):
        self.nodes = nodes

    def get_parents(self, node_name: str) -> List[str]:
        return self.nodes[node_name].parents if self.nodes[node_name].parents else []
    
    def sample_node(self, node_name: str, size: int = 1) -> np.ndarray:
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} not found in the inference object.")
        
        node = self.nodes[node_name]
        if not node.parents:
            if isinstance(node, CategoricalNode):
                return node.sample(size)
            elif node.distribution is not None:
                params = node.params if node.params is not None else {}
                samples = node.distribution.rvs(size=size, **params)
                return node.apply_inverse_transform(samples)
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
                return node.apply_inverse_transform(samples)
            
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
