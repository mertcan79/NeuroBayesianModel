import json
import os
from typing import Dict, Any, Tuple, Callable, List, Optional
from datetime import datetime
from functools import lru_cache
import logging

import pandas as pd
import numpy as np
from scipy.stats import norm, multivariate_normal
import networkx as nx
import statsmodels.api as sm
from scipy.stats import chi2_contingency

from bayesian_node import BayesianNode, CategoricalNode, Node
from structure_learning import learn_structure
from parameter_fitting import fit_parameters
from inference import Inference

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class BayesianNetwork:
    def __init__(self, method="nsl", max_parents=4, iterations=300, categorical_columns=None):
        self.method = method
        self.max_parents = max_parents
        self.iterations = iterations
        self.categorical_columns = categorical_columns or []
        self.nodes = {}
        self.edges = []
        self.data = None
        self.parameters = {}
        self.inference = None

    def fit(self, data: pd.DataFrame, prior_edges=None):
        self.data = self.preprocess_data(data)
        self.nodes = self._create_nodes_from_data(data)
        self._learn_structure(data, prior_edges)
        self._initialize_parameters(data)
        self._fit_nodes(data)
        self.inference = Inference(nodes=self.nodes)

    def _fit_nodes(self, data: pd.DataFrame):
        for node_name, node in self.nodes.items():
            parents = self.get_parents(node_name)
            if parents:
                parent_data = data[parents]
                node.fit(data[node_name], parent_data)
            else:
                node.fit(data[node_name])

    def add_node(self, node: 'BayesianNode'):
        self.nodes[node.name] = node


    def remove_edge(self, parent, child):
        self.edges.remove((parent, child))
        self.nodes[child].parents.remove(self.nodes[parent])
        self.nodes[parent].children.remove(self.nodes[child])

    def get_parents(self, node: str) -> List[str]:
        return [edge[0] for edge in self.edges if edge[1] == node]

    def get_children(self, node: str) -> List[str]:
        return [edge[1] for edge in self.edges if edge[0] == node]
    
    def get_node(self, node_name: str) -> 'BayesianNode':
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} not found in the network.")
        return self.nodes[node_name]

    def sample(self, node_name: str, size: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
        try:
            node = self.get_node(node_name)
            return node.sample(size, parent_values)
        except KeyError:
            print(f"Error: Node {node_name} not found in the network")
            return None
        except Exception as e:
            print(f"Error sampling from node {node_name}: {str(e)}")
            return None

    def _learn_structure(self, data: pd.DataFrame, prior_edges=None):

        learned_edges = learn_structure(data, method=self.method, max_parents=self.max_parents, 
                                        iterations=self.iterations, prior_edges=prior_edges)
        
        self.edges = []  # Clear existing edges
        for parent, child in learned_edges:
            if parent in self.nodes and child in self.nodes:
                self.add_edge(parent, child)
            else:
                print(f"Warning: Edge {parent} -> {child} references non-existent node")

    def add_edge(self, parent_name: str, child_name: str):
        if parent_name not in self.nodes or child_name not in self.nodes:
            raise ValueError("Parent or child node not found in the network")
        parent_node = self.nodes[parent_name]
        child_node = self.nodes[child_name]
        parent_node.add_child(child_node)
        child_node.add_parent(parent_node)
        if (parent_name, child_name) not in self.edges:
            self.edges.append((parent_name, child_name))

    def _create_nodes_from_data(self, data: pd.DataFrame) -> Dict[str, BayesianNode]:
        nodes = {}
        for column in data.columns:
            if column in self.categorical_columns:
                categories = data[column].unique().tolist()
                nodes[column] = CategoricalNode(name=column, categories=categories)
            else:
                nodes[column] = BayesianNode(name=column)
        print(f"Created {len(nodes)} nodes in total")
        return nodes

    def compute_node_influence(self, target_node_name: str) -> Dict[str, float]:
        if target_node_name not in self.nodes:
            raise ValueError(f"Node {target_node_name} not found in the network.")

        influences = {}
        for node_name in self.nodes:
            if node_name == target_node_name:
                continue
            
            # Compute influence based on correlation with the target node
            correlation = np.corrcoef(self.data[target_node_name], self.data[node_name])[0, 1]
            influences[node_name] = correlation
        
        return influences

    def compute_edge_probability(self, parent_name: str, child_name: str) -> float:
        if parent_name not in self.nodes or child_name not in self.nodes:
            raise ValueError("Parent or child node not found in the network")
        
        # Ensure the edge exists in the network
        if (parent_name, child_name) not in self.edges:
            raise ValueError(f"No edge exists from {parent_name} to {child_name}")

        parent_data = self.data[parent_name]
        child_data = self.data[child_name]
        
        # Compute edge probability based on mutual information
        contingency_table = pd.crosstab(parent_data, child_data)
        _, p_value, _, _ = chi2_contingency(contingency_table, correction=False)
        edge_probability = 1 - p_value

        return edge_probability

    def get_clinical_implications(self) -> Dict[str, Any]:
        implications = {}
        for node_name, node in self.nodes.items():
            if node.is_categorical:
                implications[node_name] = {
                    'categories': node.categories,
                    'probabilities': node.distribution
                }
            else:
                implications[node_name] = {
                    'mean': node.distribution.mean,
                    'std': node.distribution.std
                }
        return implications

    def _learn_structure(self, data: pd.DataFrame, prior_edges=None):
        learned_edges = learn_structure(data, method=self.method, max_parents=self.max_parents, 
                                        iterations=self.iterations, prior_edges=prior_edges)

        
        if not learned_edges:
            print("Warning: No edges learned. Check your structure learning method and data.")
            return
        
        self.edges = []  # Clear existing edges
        for parent, child in learned_edges:
            if parent in self.nodes and child in self.nodes:
                self.add_edge(parent, child)
            else:
                print(f"Warning: Edge {parent} -> {child} references non-existent node")
        

    def _initialize_parameters(self, data: pd.DataFrame, prior: Dict[str, Any] = None):
        if prior is None:
            prior = {}
        
        for node_name, node in self.nodes.items():
            parents = self.get_parents(node_name)

            if node_name in self.categorical_columns:
                node.set_categorical(data[node_name].unique())
                counts = data[node_name].value_counts()
                self.parameters[node_name] = {
                    'categories': counts.index.tolist(),
                    'probabilities': (counts / counts.sum()).to_dict()
                }
                node.distribution = self.parameters[node_name]['probabilities']
            else:
                # Continuous node
                if not parents:
                    self.parameters[node_name] = {
                        'mean': data[node_name].mean(),
                        'std': data[node_name].std()
                    }
                    node.distribution = norm(loc=self.parameters[node_name]['mean'], 
                                            scale=self.parameters[node_name]['std'])
                else:
                    X = data[parents]
                    y = data[node_name]
                    X = sm.add_constant(X)
                    model = sm.OLS(y, X).fit()
                    self.parameters[node_name] = {
                        'coefficients': model.params.values,
                        'std': model.resid.std()
                    }
                    node.distribution = norm(loc=0, scale=self.parameters[node_name]['std'])

            if parents:
                if node.is_categorical:
                    # For categorical nodes with parents, we'll use a simple conditional probability table
                    self.parameters[node_name]['cpt'] = {}
                    for parent_values, group in data.groupby(parents):
                        counts = group[node_name].value_counts()
                        probs = counts / counts.sum()
                        self.parameters[node_name]['cpt'][parent_values] = probs.to_dict()
                else:
                    # The distribution for continuous nodes with parents is already set above
                    pass

    def _expectation_step(self, data: pd.DataFrame):
        responsibilities = {}
        for node in self.nodes:
            parents = self.get_parents(node)
            node_data = data[node]
            parent_data = data[parents] if parents else None
            mean = node_data.mean() if node_data.size else 0
            std = node_data.std() if node_data.size else 1e-6
            responsibilities[node] = {
                'mean': mean,
                'std': std,
                'parent_data': parent_data  # Store parent data for later use
            }
        return responsibilities

    def _maximization_step(self, data: pd.DataFrame, responsibilities: Dict[str, Any]):
        for node in self.nodes:
            if node in responsibilities:
                mean = responsibilities[node]['mean']
                std = responsibilities[node]['std']
                self.parameters[node] = {
                    'mean': mean,
                    'std': std
                }

    def _compute_log_likelihood(self, data: pd.DataFrame) -> float:
        log_likelihood = 0
        for node in self.nodes:
            parents = self.get_parents(node)
            node_data = data[node]
            if parents:
                parent_data = data[parents]
                mean = self._predict_mean(node, parent_data)
                std = self.parameters[node]['std']
            else:
                mean = self.parameters[node]['mean']
                std = self.parameters[node]['std']
            log_likelihood += norm.logpdf(node_data, loc=mean, scale=std).sum()
        return log_likelihood

    def _predict_mean(self, node: str, parent_data: pd.DataFrame) -> np.ndarray:
        coeff = self.parameters[node]['coefficients']
        X = sm.add_constant(parent_data)
        return X @ coeff

    def save(self, file_path: str):
        network_data = {
            'nodes': {name: node.to_dict() for name, node in self.nodes.items()},
            'edges': self.edges,
            'parameters': self.parameters,
            'categorical_columns': self.categorical_columns
        }
        with open(file_path, 'w') as f:
            json.dump(network_data, f, indent=2)

    @classmethod
    def load(cls, file_path: str) -> 'BayesianNetwork':
        with open(file_path, 'r') as f:
            network_data = json.load(f)
        
        instance = cls(categorical_columns=network_data.get('categorical_columns', []))
        for name, node_data in network_data['nodes'].items():
            if name in instance.categorical_columns:
                node = CategoricalNode(name=name)
            else:
                node = BayesianNode(name=name)
            node.from_dict(node_data)
            instance.nodes[name] = node
        
        instance.edges = network_data['edges']
        instance.parameters = network_data['parameters']
        return instance

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.fillna(data.mean())  # Fill NaNs with mean values
        data = data.replace([np.inf, -np.inf], np.nan).dropna()  # Remove inf values
        for col in self.categorical_columns:
            data[col] = data[col].astype('category')
        return data

