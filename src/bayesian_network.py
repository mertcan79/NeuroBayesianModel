import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Callable, List
from sklearn.impute import SimpleImputer
import logging
from sklearn.model_selection import KFold
from scipy import stats
import copy

from bayesian_node import BayesianNode, CategoricalNode
from structure_learning import learn_structure
from parameter_fitting import fit_parameters
from inference import log_likelihood, sample_node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesianNetwork:
    def __init__(self, method='k2', max_parents=5, categorical_columns=None):
        self.method = method
        self.max_parents = max_parents
        self.nodes = {}
        self.categorical_columns = categorical_columns or []
        self.prior_edges = {}

    def fit(self, data: pd.DataFrame, prior_edges: List[tuple] = None, progress_callback: Callable[[float], None] = None):
        preprocessed_data = self.preprocess_data(data)
        try:
            logger.info("Learning structure")
            self._create_nodes(preprocessed_data)
            self._learn_structure(preprocessed_data, prior_edges)
            if progress_callback:
                progress_callback(0.5)
            logger.info("Fitting parameters")
            fit_parameters(self.nodes, preprocessed_data)
            if progress_callback:
                progress_callback(1.0)
        except Exception as e:
            logger.error(f"Error during fitting: {str(e)}")
            raise

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        for col in self.categorical_columns:
            data[col] = data[col].astype('category').cat.codes
        
        return data

    def _learn_structure(self, data: pd.DataFrame, prior_edges: List[tuple] = None):
        self.nodes = learn_structure(data, method=self.method, max_parents=self.max_parents, prior_edges=self.prior_edges)
        if prior_edges:
            for edge in prior_edges:
                if edge[1] not in self.nodes:
                    self.nodes[edge[1]] = BayesianNode(edge[1])
                if edge[0] not in self.nodes:
                    self.nodes[edge[0]] = BayesianNode(edge[0])
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

    def sample_node(self, node_name: str, size: int = 1) -> np.ndarray:
        node = self.nodes[node_name]
        if not node.parents:
            if isinstance(node, CategoricalNode):
                return node.sample(size)
            elif callable(node.distribution):
                try:
                    samples = node.distribution.rvs(size=size, **node.params)
                except AttributeError:
                    samples = np.array([node.distribution(**node.params) for _ in range(size)])
                return node.inverse_transform(samples)
            else:
                raise ValueError(f"Unsupported distribution type for node {node_name}")
        else:
            parent_values = np.column_stack([self.sample_node(p.name, size) for p in node.parents])
            if isinstance(node, CategoricalNode):
                return node.sample(size)
            else:
                beta = node.params.get('beta', np.zeros(len(node.parents)))
                intercept = node.params.get('intercept', 0)
                scale = node.params.get('scale', 1.0)
                loc = intercept + np.dot(parent_values, beta)
                noise = np.random.normal(0, scale, size)
                samples = loc + noise
                return node.inverse_transform(samples)

    def cross_validate(self, data: pd.DataFrame, k_folds: int = 5) -> Tuple[float, float]:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        log_likelihoods = []
        for train_index, test_index in kf.split(data):
            train_data, test_data = data.iloc[train_index], data.iloc[test_index]
            fold_bn = BayesianNetwork(method=self.method, max_parents=self.max_parents)
            fold_bn.fit(train_data)
            log_likelihoods.append(fold_bn.log_likelihood(test_data))
        return float(np.mean(log_likelihoods)), float(np.std(log_likelihoods))

    def compute_sensitivity(self, target_node: str, num_samples: int = 10000) -> Dict[str, float]:
        sensitivity = {}
        base_samples = self.sample_node(target_node, size=num_samples)
        logger.info(f"Base samples for {target_node}: {base_samples[:5]}")  # Log first 5 samples
        
        for node_name, node in self.nodes.items():
            if node_name != target_node:
                perturbed_network = copy.deepcopy(self)
                perturbed_samples = perturbed_network.sample_node(node_name, size=num_samples)
                
                logger.info(f"Perturbed samples for {node_name}: {perturbed_samples[:5]}")  # Log first 5 samples
                
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
                
                logger.info(f"Perturbed output for {target_node}: {perturbed_output[:5]}")  # Log first 5 samples
                
                if isinstance(self.nodes[target_node], CategoricalNode):
                    sensitivity[node_name] = np.mean(perturbed_output != base_samples)
                else:
                    sensitivity[node_name] = np.mean(np.abs(perturbed_output - base_samples))
        
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
