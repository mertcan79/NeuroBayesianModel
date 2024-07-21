import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Callable, List
from sklearn.impute import SimpleImputer
import logging
from sklearn.model_selection import KFold
from scipy import stats
import copy
from collections import deque
from bayesian_node import BayesianNode, CategoricalNode
from structure_learning import learn_structure
from parameter_fitting import fit_parameters
from inference import log_likelihood, sample_node
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import networkx as nx
import json
import os
from datetime import datetime
from bayesian_node import BayesianNode, CategoricalNode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesianNetwork:
    def __init__(self, method='k2', max_parents=5, categorical_columns=None):
        self.method = method
        self.max_parents = max_parents
        self.nodes = {}
        self.categorical_columns = categorical_columns or []
        self.prior_edges = {}

    def sample_node(self, node_name: str, size: int = 1) -> np.ndarray:
        sorted_nodes = self.topological_sort()
        samples = {}
        for node in sorted_nodes:
            if node == node_name:
                break
            samples[node] = self.nodes[node].sample(size, samples)
        return self.nodes[node_name].sample(size, samples)

    def fit(self, data: pd.DataFrame, prior_edges: List[tuple] = None, progress_callback: Callable[[float], None] = None):
        try:
            logger.info("Learning structure")
            self._learn_structure(data, prior_edges)
            if progress_callback:
                progress_callback(0.5)
            logger.info("Fitting parameters")
            fit_parameters(self.nodes, data)
            if progress_callback:
                progress_callback(1.0)
        except Exception as e:
            logger.error(f"Error during fitting: {str(e)}")
            raise

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        # Convert categorical columns
        for col in self.categorical_columns:
            data[col] = pd.Categorical(data[col]).codes

        # Separate numeric and categorical columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        return data

    def _learn_structure(self, data: pd.DataFrame, prior_edges: List[tuple] = None):
        self.nodes = learn_structure(data, method=self.method, max_parents=self.max_parents, 
                                    prior_edges=self.prior_edges, categorical_columns=self.categorical_columns)
        if prior_edges:
            for edge in prior_edges:
                if edge[0] in self.nodes and edge[1] in self.nodes:
                    if self.nodes[edge[0]] not in self.nodes[edge[1]].parents:
                        self.nodes[edge[1]].parents.append(self.nodes[edge[0]])
                        self.nodes[edge[0]].children.append(self.nodes[edge[1]])

    def _create_nodes(self, data: pd.DataFrame):
        for column in data.columns:
            if column in self.categorical_columns:
                categories = data[column].astype('category').cat.categories.tolist()
                self.nodes[column] = CategoricalNode(column, categories)
            else:
                self.nodes[column] = BayesianNode(column)

    def log_likelihood(self, data: pd.DataFrame) -> float:
        return log_likelihood(self.nodes, data)

    def cross_validate(self, data: pd.DataFrame, k_folds: int = 5) -> Tuple[float, float]:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        log_likelihoods = []
        
        for train_index, test_index in kf.split(data):
            train_data, test_data = data.iloc[train_index], data.iloc[test_index]
            fold_bn = BayesianNetwork(method=self.method, max_parents=self.max_parents, categorical_columns=self.categorical_columns)
            fold_bn.fit(train_data)
            log_likelihoods.append(fold_bn.log_likelihood(test_data))
        
        return float(np.mean(log_likelihoods)), float(np.std(log_likelihoods))

    def compute_sensitivity(self, target_node: str, num_samples: int = 10000) -> Dict[str, float]:
        sensitivity = {}
        base_samples = self.sample_node(target_node, size=num_samples)
        
        for node_name, node in self.nodes.items():
            if node_name != target_node:
                # Node value sensitivity
                perturbed_network = copy.deepcopy(self)
                perturbed_samples = perturbed_network.sample_node(node_name, size=num_samples)
                
                if isinstance(node, CategoricalNode):
                    unique, counts = np.unique(perturbed_samples, return_counts=True)
                    prob_dict = dict(zip(unique, counts / len(perturbed_samples)))
                    probabilities = [prob_dict.get(cat, 0) for cat in node.categories]
                    perturbed_network.nodes[node_name].set_distribution(probabilities)
                else:
                    perturbed_samples_transformed = node.transform(perturbed_samples)
                    mean = np.mean(perturbed_samples_transformed)
                    std = np.std(perturbed_samples_transformed)
                    perturbed_network.nodes[node_name].params = {'loc': mean, 'scale': std}
                    perturbed_network.nodes[node_name].distribution = stats.norm
                
                perturbed_output = perturbed_network.sample_node(target_node, size=num_samples)
                
                if isinstance(self.nodes[target_node], CategoricalNode):
                    sensitivity[node_name] = np.mean(perturbed_output != base_samples)
                else:
                    sensitivity[node_name] = np.mean(np.abs(perturbed_output - base_samples))

                # Parameter sensitivity
                for param, value in node.params.items():
                    perturbed_network = copy.deepcopy(self)
                    perturbed_param = value * 1.1  # 10% increase
                    perturbed_network.nodes[node_name].params[param] = perturbed_param
                    perturbed_output = perturbed_network.sample_node(target_node, size=num_samples)
                    
                    if isinstance(self.nodes[target_node], CategoricalNode):
                        sensitivity[f"{node_name}_{param}"] = np.mean(perturbed_output != base_samples)
                    else:
                        sensitivity[f"{node_name}_{param}"] = np.mean(np.abs(perturbed_output - base_samples))
        
        # Normalize sensitivity scores
        max_sensitivity = max(sensitivity.values())
        if max_sensitivity > 0:
            sensitivity = {k: v / max_sensitivity for k, v in sensitivity.items()}
        
        return sensitivity

    def metropolis_hastings(self, observed_data: Dict[str, Any], num_samples: int = 1000) -> Dict[str, List[Any]]:
        current_state = {node: self.sample_node(node, size=1)[0] for node in self.nodes}
        samples = {node: [] for node in self.nodes}
        for _ in range(num_samples):
            proposed_state = current_state.copy()
            node_to_change = np.random.choice(list(self.nodes.keys()))
            proposed_state[node_to_change] = self.sample_node(node_to_change, size=1)[0]
            current_likelihood = self.log_likelihood(pd.DataFrame([current_state]))
            proposed_likelihood = self.log_likelihood(pd.DataFrame([proposed_state]))
            if np.log(np.random.random()) < proposed_likelihood - current_likelihood:
                current_state = proposed_state
            for node, value in current_state.items():
                samples[node].append(value)
        return samples
        
    def explain_structure(self) -> Dict[str, List[str]]:
        structure = {}
        for node_name, node in self.nodes.items():
            structure[node_name] = [parent.name for parent in node.parents]
        return structure
    
    def set_distribution(self, distribution, params=None):
        self.distribution = distribution
        self.params = params or {}
        # Validate probabilities if using categorical distributions
        if isinstance(self, CategoricalNode):
            probs = self.params.get('p')
            if probs is not None:
                if not np.all((0 <= probs) & (probs <= 1)):
                    raise ValueError("Probabilities must be between 0 and 1.")
                if not np.isclose(np.sum(probs), 1):
                    raise ValueError("Probabilities must sum to 1.")

    def topological_sort(self):
        visited = set()
        stack = deque()

        def visit(node):
            if node.name not in visited:
                visited.add(node.name)
                for child in node.children:
                    visit(child)
                stack.appendleft(node.name)

        for node in self.nodes.values():
            visit(node)

        return list(stack)
    
    def tune_hyperparameters(self, data: pd.DataFrame, param_grid: Dict[str, List[Any]]):
        def score_model(max_parents):
            self.max_parents = max_parents
            self.fit(data)
            return self.log_likelihood(data)

        grid_search = GridSearchCV(estimator=self, param_grid=param_grid, scoring=score_model, cv=5)
        grid_search.fit(data)
        self.max_parents = grid_search.best_params_['max_parents']
        
    def compare_structures(self, data: pd.DataFrame, structures: List[Dict[str, List[str]]]):
        results = []
        for structure in structures:
            bn = BayesianNetwork(method='custom', max_parents=self.max_parents)
            bn.nodes = {node: BayesianNode(node) for node in structure.keys()}
            for node, parents in structure.items():
                bn.nodes[node].parents = [bn.nodes[p] for p in parents]
            bn.fit(data)
            ll = bn.log_likelihood(data)
            bic = -2 * ll + np.log(len(data)) * sum(len(parents) for parents in structure.values())
            results.append({'structure': structure, 'log_likelihood': ll, 'BIC': bic})
        return sorted(results, key=lambda x: x['BIC'])
    
    def explain_structure_extended(self) -> Dict[str, Any]:
        G = nx.DiGraph()
        for node_name, node in self.nodes.items():
            G.add_node(node_name)
            for parent in node.parents:
                G.add_edge(parent.name, node_name)

        pos = nx.spring_layout(G)
        
        structure = {
            "nodes": [
                {
                    "id": node,
                    "label": node,
                    "x": pos[node][0],
                    "y": pos[node][1],
                    "attributes": {
                        "distribution": str(self.nodes[node].distribution),
                        "params": self.nodes[node].params
                    }
                } for node in G.nodes()
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "weight": self.nodes[v].params.get('beta', [1])[self.nodes[v].parents.index(self.nodes[u])]
                            if u in [p.name for p in self.nodes[v].parents] else 1
                } for u, v in G.edges()
            ],
            "global_stats": {
                "num_nodes": len(G.nodes()),
                "num_edges": len(G.edges()),
                "avg_degree": sum(dict(G.degree()).values()) / len(G),
                "clustering_coefficient": nx.average_clustering(G),
                "avg_path_length": nx.average_shortest_path_length(G) if nx.is_connected(G) else None
            }
        }
        return structure
    
    def write_results_to_json(self, results: Dict[str, Any], filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"

        log_folder = "logs"
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        file_path = os.path.join(log_folder, filename)

        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results written to {file_path}")
            
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
            if level_constraints:
                allowed_parents += level_constraints.get(level, [])
            self._learn_structure(level_data, allowed_parents=allowed_parents)
        fit_parameters(self.nodes, preprocessed_data)