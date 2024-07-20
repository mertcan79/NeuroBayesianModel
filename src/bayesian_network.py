from bayesian_node import BayesianNode, CategoricalNode
from structure_learning import learn_structure
from parameter_fitting import fit_parameters
from inference import log_likelihood, sample_node
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from sklearn.impute import SimpleImputer
import logging
from sklearn.model_selection import KFold
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
import copy
from data_processing import preprocess_data
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesianNetwork:
    def __init__(self, method='k2', max_parents=5):
        self.method = method
        self.max_parents = max_parents
        self.nodes: Dict[str, BayesianNode] = {}
        self.imputer = SimpleImputer(strategy='mean')
        self.prior_edges = {}
        self.categorical_columns = []

    def copy(self):
        new_network = BayesianNetwork(method=self.method, max_parents=self.max_parents)
        new_network.nodes = {name: copy.deepcopy(node) for name, node in self.nodes.items()}
        new_network.imputer = copy.deepcopy(self.imputer)
        new_network.prior_edges = self.prior_edges.copy()
        return new_network

    def set_edge_prior(self, parent: str, child: str, probability: float):
        self.prior_edges[(parent, child)] = probability
        
    def set_categorical_columns(self, categorical_columns: List[str]):
        self.categorical_columns = categorical_columns

    def fit(self, data: pd.DataFrame, prior_edges: List[tuple] = None, progress_callback: Callable[[float], None] = None):
        preprocessed_data = preprocess_data(data)
        try:
            logger.info("Starting to learn structure")
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

    def infer_with_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        results = pd.DataFrame()
        for _, row in data.iterrows():
            observed = {col: val for col, val in row.items() if pd.notna(val)}
            sampled = self.metropolis_hastings(observed, num_samples=1000)
            results = results.append(sampled, ignore_index=True)
        return results

    def feature_importance(self, target_node: str) -> Dict[str, float]:
        sensitivity = self.sensitivity_analysis(target_node)
        total = sum(sensitivity.values())
        return {node: value/total for node, value in sensitivity.items() if total != 0}

    def _learn_structure(self, data: pd.DataFrame, prior_edges: List[tuple] = None):
        imputed_data = self.imputer.fit_transform(data)
        imputed_data = pd.DataFrame(imputed_data, columns=data.columns)

        self.nodes = learn_structure(imputed_data, method=self.method, max_parents=self.max_parents, prior_edges=self.prior_edges)

        if prior_edges:
            for edge in prior_edges:
                if edge not in self.nodes[edge[1]].parents:
                    self.nodes[edge[1]].parents.append(self.nodes[edge[0]])
                    self.nodes[edge[0]].children.append(self.nodes[edge[1]])

        logger.info("Learned graph structure:")
        for node_name, node in self.nodes.items():
            logger.info(f"{node_name} -> parents: {[p.name for p in node.parents]}, children: {[c.name for c in node.children]}")

    def log_likelihood(self, data: pd.DataFrame) -> float:
        return log_likelihood(self.nodes, data)

    def sample_node(self, node_name: str, size: int = 1) -> np.ndarray:
        print(f"sample_node called with node_name: {node_name}, size: {size}")
        node = self.nodes[node_name]
        if not node.parents:
            if isinstance(node, CategoricalNode):
                samples = node.sample(size)
            elif callable(node.distribution):
                try:
                    samples = node.distribution.rvs(size=size)
                except AttributeError:
                    samples = np.array([node.distribution() for _ in range(size)])
            else:
                raise ValueError(f"Unsupported distribution type for node {node_name}")
        else:
            parent_values = np.column_stack([self.sample_node(p.name, size) for p in node.parents])
            if isinstance(node, CategoricalNode):
                samples = node.sample(size)
            else:
                loc = np.dot(parent_values, node.params.get('beta', np.zeros(parent_values.shape[1])))
                scale = node.params.get('scale', 1.0)
                samples = node.distribution.rvs(loc=loc, scale=scale, size=size)
        
        return np.array(samples).reshape(-1)


    def cross_validate(self, data: pd.DataFrame, k_folds: int = 5) -> Tuple[float, float]:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        log_likelihoods = []

        for i, (train_index, test_index) in enumerate(kf.split(data)):
            logger.info(f"Starting fold {i+1}/{k_folds}")
            train_data, test_data = data.iloc[train_index], data.iloc[test_index]
            
            # Create a new instance of BayesianNetwork for each fold
            fold_bn = BayesianNetwork(method=self.method, max_parents=self.max_parents)
            fold_bn.fit(train_data)
            
            log_likelihood = fold_bn.log_likelihood(test_data)
            log_likelihoods.append(log_likelihood)
            logger.info(f"Completed fold {i+1}/{k_folds}, log-likelihood: {log_likelihood}")

        return float(np.mean(log_likelihoods)), float(np.std(log_likelihoods))

    def _create_nodes(self, data: pd.DataFrame):
        for column in data.columns:
            if column in self.categorical_columns:
                categories = data[column].unique().tolist()
                self.nodes[column] = CategoricalNode(column, categories)
            else:
                self.nodes[column] = BayesianNode(column)

    def explain_structure(self) -> Dict[str, List[str]]:
        return {node: [parent.name for parent in self.nodes[node].parents] for node in self.nodes}
    
    def explain_prediction(self, node: str, observation: Dict[str, Any]) -> str:
        parents = self.nodes[node].parents
        if not parents:
            return f"{node} is a root node and doesn't depend on other variables."
        
        explanation = f"{node} is influenced by: "
        for parent in parents:
            parent_value = observation.get(parent.name, "Unknown")
            explanation += f"\n- {parent.name} (value: {parent_value})"
        
        return explanation   

    def marginal_effect(self, target_node: str, input_node: str, num_samples: int = 1000) -> pd.DataFrame:
        input_values = np.linspace(self.nodes[input_node].distribution.ppf(0.01),
                                   self.nodes[input_node].distribution.ppf(0.99),
                                   num=20)
        effects = []
        for value in input_values:
            samples = self.sample_node(target_node, size=num_samples, conditions={input_node: value})
            effects.append(np.mean(samples))
        
        return pd.DataFrame({input_node: input_values, f"Effect on {target_node}": effects})

    
    def compute_sensitivity(self, target_node: str, num_samples: int = 10000) -> Dict[str, float]:
        sensitivity = {}
        base_samples = self.sample_node(target_node, size=num_samples)
        
        for node_name, node in self.nodes.items():
            if node_name != target_node:
                try:
                    perturbed_network = self.copy()
                    perturbed_samples = perturbed_network.sample_node(node_name, size=num_samples)
                    
                    if isinstance(node, BayesianNode):  # Continuous node
                        kde = stats.gaussian_kde(perturbed_samples)
                        perturbed_network.nodes[node_name].distribution = lambda size: kde.resample(size)[0]
                    elif isinstance(node, CategoricalNode):  # Categorical node
                        counts = np.bincount(perturbed_samples, minlength=len(node.categories))
                        perturbed_network.nodes[node_name].set_distribution(counts)
                    
                    perturbed_output = perturbed_network.sample_node(target_node, size=num_samples)
                    sensitivity[node_name] = np.mean(np.abs(perturbed_output - base_samples))
                except Exception as e:
                    logger.error(f"Error in sensitivity analysis for node {node_name}: {str(e)}")
                    sensitivity[node_name] = np.nan
        
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
                if node not in observed_data:
                    samples[node].append(value)

        return samples

    def sensitivity_analysis(self, target_node: str, n_samples: int = 1000) -> Dict[str, float]:
        results = {}
        for node_name in self.nodes:
            if node_name != target_node:
                try:
                    print(f"Calling sample_node for {target_node} with {n_samples} samples")
                    original_samples = self.sample_node(target_node, size=n_samples)
                    
                    perturbed_network = self.copy()
                    print(f"Calling sample_node for {node_name} with {n_samples} samples")
                    perturbed_samples = perturbed_network.sample_node(node_name, size=n_samples)
                    
                    if isinstance(perturbed_network.nodes[node_name], CategoricalNode):
                        counts = np.bincount(perturbed_samples, minlength=len(perturbed_network.nodes[node_name].categories))
                        perturbed_network.nodes[node_name].set_distribution(counts)
                    else:
                        kde = stats.gaussian_kde(perturbed_samples)
                        perturbed_network.nodes[node_name].distribution = lambda size: kde.resample(size)[0]
                    
                    new_samples = perturbed_network.sample_node(target_node, size=n_samples)
                    sensitivity = np.mean(np.abs(new_samples - original_samples))
                    results[node_name] = sensitivity
                except Exception as e:
                    logging.error(f"Error in sensitivity analysis for node {node_name}: {str(e)}")
        return results
    
    def train(self, data, iterations):
        for _ in tqdm(range(iterations), desc="Training Network"):
            self.fit(data)
            
    def k_fold_cross_validation(network, data, k=5):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        log_likelihoods = []

        for train_index, test_index in kf.split(data):
            train_data, test_data = data.iloc[train_index], data.iloc[test_index]
            network.fit(train_data)
            log_likelihood = network.log_likelihood(test_data)
            log_likelihoods.append(log_likelihood)

        return np.mean(log_likelihoods), np.std(log_likelihoods)