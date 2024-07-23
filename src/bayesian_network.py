import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Callable, List
import logging
from sklearn.model_selection import KFold
from scipy import stats
import copy
from collections import deque
from bayesian_node import BayesianNode, CategoricalNode
from structure_learning import learn_structure
from parameter_fitting import fit_parameters
import networkx as nx
import json
import os
from datetime import datetime
from functools import lru_cache
from joblib import Parallel, delayed
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesianNetwork:
    def __init__(self, method='hill_climb', max_parents=3, iterations=100, categorical_columns=None):
        self.method = method
        self.max_parents = max_parents
        self.iterations = iterations
        self.nodes = {}
        self.categorical_columns = categorical_columns or []
        self.prior_edges = {}
        
        for col in self.categorical_columns:
            self.nodes[col] = CategoricalNode(col, [])  # Empty categories for now

    def to_dict(self):
        return {
            'nodes': {name: node.to_dict() for name, node in self.nodes.items()},
            'method': self.method,
            'max_parents': self.max_parents,
            'categorical_columns': self.categorical_columns
        }
        
    @classmethod
    def from_dict(cls, data):
        bn = cls(method=data['method'], max_parents=data['max_parents'], categorical_columns=data['categorical_columns'])
        bn.nodes = {}
        for name, node_data in data['nodes'].items():
            if name in bn.categorical_columns:
                bn.nodes[name] = CategoricalNode.from_dict(node_data)
            else:
                bn.nodes[name] = BayesianNode.from_dict(node_data)
        
        # Reconstruct parent-child relationships
        for name, node in bn.nodes.items():
            node.parents = [bn.nodes[parent_name] for parent_name in node_data['parents']]
            node.children = [bn.nodes[child_name] for child_name in node_data['children']]
        
        return bn

    def write_results_to_json(self, results: Dict[str, Any], filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"

        log_folder = "logs"
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        file_path = os.path.join(log_folder, filename)

        # Ensure all values are JSON serializable
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            else:
                return obj

        # Include results from explain_structure_extended
        network_structure = self.explain_structure_extended()
        results["network_structure"] = network_structure

        # Include results from HierarchicalBayesianNetwork if available
        if isinstance(self, HierarchicalBayesianNetwork):
            results["hierarchical_structure"] = self.explain_hierarchical_structure()

        serializable_results = make_serializable(results)

        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results written to {file_path}")
        
    def get_edges(self):
        edges = []
        for node_name, node in self.nodes.items():
            for parent in node.parents:
                edges.append((parent.name, node_name))
        return edges   
        
    @lru_cache(maxsize=128)
    def sample_node(self, node_name: str, size: int = 1) -> np.ndarray:
        sorted_nodes = self.topological_sort()
        samples = {node: None for node in sorted_nodes}
        
        for node in sorted_nodes:
            if node == node_name:
                break
            parent_values = {parent: samples[parent] for parent in self.nodes[node].parents}
            samples[node] = self.nodes[node].sample(size, parent_values)
            
        return samples[node_name]

    def fit(self, data: pd.DataFrame, prior_edges: List[tuple] = None, progress_callback: Callable[[float], None] = None):
        try:
            for col in self.categorical_columns:
                if col in self.nodes:
                    self.nodes[col].categories = sorted(data[col].unique())

            logger.info("Learning structure")
            self._learn_structure(data)  # Remove prior_edges from here
            if progress_callback:
                progress_callback(0.5)
            
            logger.info("Fitting parameters")
            self._fit_parameters(data)
            if progress_callback:
                progress_callback(1.0)
        except ValueError as ve:
            logger.error(f"ValueError during fitting: {str(ve)}")
            raise
        except KeyError as ke:
            logger.error(f"KeyError during fitting: {str(ke)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during fitting: {str(e)}")
            raise

    def set_parameters(self, node_name, values, parent_variables):
        self.nodes[node_name].set_parameters(values, parent_variables)

    def _fit_parameters(self, data: pd.DataFrame):
        for node_name, node in self.nodes.items():
            parent_names = [parent.name for parent in node.parents]
            node_data = data[node_name]
            parent_data = data[parent_names] if parent_names else None
            node.fit(node_data, parent_data)

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.apply(lambda col: pd.Categorical(col).codes if col.name in self.categorical_columns else col)

    def _learn_structure(self, data: pd.DataFrame):
        self.nodes = learn_structure(
            data, 
            method=self.method, 
            max_parents=self.max_parents, 
            iterations=self.iterations,
            prior_edges=self.prior_edges
        )

    def _create_nodes(self, data: pd.DataFrame):
        for column in data.columns:
            if column in self.categorical_columns:
                categories = data[column].astype('category').cat.categories.tolist()
                self.nodes[column] = CategoricalNode(column, categories)
            else:
                self.nodes[column] = BayesianNode(column)

    def add_edge(self, parent, child):
        if parent not in self.nodes or child not in self.nodes:
            raise ValueError("Both nodes must exist in the network")
        self.nodes[child].parents.append(self.nodes[parent])
        self.nodes[parent].children.append(self.nodes[child])

    @lru_cache(maxsize=128)
    def _cached_node_log_likelihood(self, node_name, value, parent_values):
        node = self.nodes[node_name]
        return node.log_probability(value, parent_values)

    def log_likelihood(self, data: pd.DataFrame) -> float:
        log_probs = np.zeros(len(data))
        for node_name, node in self.nodes.items():
            parent_names = [parent.name for parent in node.parents]
            if parent_names:
                parent_values = data[parent_names].values
                log_probs += np.array([node.log_probability(val, tuple(pv)) for val, pv in zip(data[node_name], parent_values)])
            else:
                log_probs += np.array([node.log_probability(val) for val in data[node_name]])
        return np.sum(log_probs)

    def cross_validate(self, data: pd.DataFrame, k_folds: int = 5) -> Tuple[float, float]:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        log_likelihoods = Parallel(n_jobs=-1)(
            delayed(self._evaluate_fold)(train_index, test_index, data) for train_index, test_index in kf.split(data)
        )
        return float(np.mean(log_likelihoods)), float(np.std(log_likelihoods))

    def _evaluate_fold(self, train_index, test_index, data):
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]
        fold_bn = BayesianNetwork(method=self.method, max_parents=self.max_parents, categorical_columns=self.categorical_columns)
        fold_bn.fit(train_data)
        return fold_bn.log_likelihood(test_data)

    def compute_sensitivity(self, target_node: str, num_samples: int = 10000) -> Dict[str, float]:
        sensitivity = {}
        try:
            base_samples = self.sample_node(target_node, size=num_samples)
            
            for node_name, node in self.nodes.items():
                if node_name != target_node:
                    try:
                        perturbed_network = copy.deepcopy(self)
                        perturbed_samples = perturbed_network.sample_node(node_name, size=num_samples)
                        
                        if hasattr(node, 'transform'):
                            perturbed_samples_transformed = node.transform(perturbed_samples)
                        else:
                            perturbed_samples_transformed = perturbed_samples
                        
                        if isinstance(node, CategoricalNode):
                            unique, counts = np.unique(perturbed_samples, return_counts=True)
                            prob_dict = dict(zip(unique, counts / len(perturbed_samples)))
                            probabilities = [prob_dict.get(cat, 0) for cat in node.categories]
                            perturbed_network.nodes[node_name].set_distribution(probabilities)
                        else:
                            mean = np.mean(perturbed_samples_transformed)
                            std = np.std(perturbed_samples_transformed)
                            perturbed_network.nodes[node_name].set_distribution((mean, std))

                        perturbed_samples = perturbed_network.sample_node(target_node, size=num_samples)
                        sensitivity[node_name] = np.mean(np.abs(base_samples - perturbed_samples))
                    except Exception as e:
                        logger.warning(f"Could not compute sensitivity for node {node_name}: {e}")
        except Exception as e:
            logger.error(f"Error in compute_sensitivity: {e}")
        return sensitivity

    def simulate_intervention(self, interventions: Dict[str, Any], size: int = 1000) -> pd.DataFrame:
        samples = {}
        sorted_nodes = self.topological_sort()
        
        for node in sorted_nodes:
            if node in interventions:
                samples[node] = np.repeat(interventions[node], size)
            else:
                parent_values = {parent: samples[parent] for parent in self.nodes[node].parents}
                samples[node] = self.nodes[node].sample(size, parent_values)
        
        return pd.DataFrame(samples)

    def topological_sort(self) -> List[str]:
        graph = nx.DiGraph()
        for node_name, node in self.nodes.items():
            graph.add_node(node_name)
            for parent in node.parents:
                graph.add_edge(parent.name, node_name)
        
        return list(nx.topological_sort(graph))

    def explain_structure(self):
        return {
            "nodes": list(self.nodes.keys()),
            "edges": self.get_edges()
        }

    def explain_structure_extended(self):
        structure = {
            "nodes": list(self.nodes.keys()),
            "edges": self.get_edges()
        }

        for node_name, node in self.nodes.items():
            structure[node_name] = {
                "parents": [parent.name for parent in node.parents],
                "parameters": node.parameters if hasattr(node, 'parameters') else None,
                "distribution": str(node.distribution) if hasattr(node, 'distribution') else None
            }

        return structure

    def fit_transform(self, data: pd.DataFrame, prior_edges: List[tuple] = None, progress_callback: Callable[[float], None] = None):
        self.fit(data, prior_edges=prior_edges, progress_callback=progress_callback)
        return self.transform(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed_data = data.copy()
        for col in self.categorical_columns:
            if col in transformed_data:
                transformed_data[col] = transformed_data[col].astype('category').cat.codes
        return transformed_data

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

class HierarchicalBayesianNetwork(BayesianNetwork):
    def __init__(self, levels: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.levels = levels
        self.level_nodes = {level: [] for level in levels}

    def add_node(self, node: str, level: str):
        if level not in self.levels:
            raise ValueError(f"Invalid level: {level}")
        self.nodes[node] = BayesianNode(node)
        self.level_nodes[level].append(node)

    def add_edge(self, parent: str, child: str):
        if parent not in self.nodes or child not in self.nodes:
            raise ValueError("Both nodes must exist in the network")
        self.nodes[child].parents.append(self.nodes[parent])
        self.nodes[parent].children.append(self.nodes[child])

    def fit(self, data: pd.DataFrame, level_constraints: Dict[str, List[str]] = None):
        preprocessed_data = self.preprocess_data(data)
        for level in self.levels:
            level_data = preprocessed_data[self.level_nodes[level]]
            allowed_parents = self.level_nodes[level]
            if level_constraints and level in level_constraints:
                allowed_parents += level_constraints[level]
            self._learn_structure(level_data, allowed_parents=allowed_parents)
        fit_parameters(self.nodes, preprocessed_data)

    def _learn_structure(self, data: pd.DataFrame, allowed_parents: List[str]):
        self.nodes = learn_structure(data, method=self.method, max_parents=self.max_parents, 
                                    prior_edges=self.prior_edges, categorical_columns=self.categorical_columns)
        for node in self.nodes.values():
            node.parents = [parent for parent in node.parents if parent.name in allowed_parents]

        # Enforce the structure according to the allowed_parents
        for node_name, node in self.nodes.items():
            node.parents = [self.nodes[parent_name] for parent_name in allowed_parents if parent_name in self.nodes and parent_name in node.parents]