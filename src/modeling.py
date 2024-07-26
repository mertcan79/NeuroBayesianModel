import pandas as pd
import numpy as np
import logging
from typing import List, Callable, Tuple, Dict, Any
from scipy.special import logsumexp
import networkx as nx

from bayesian_node import BayesianNode, CategoricalNode
from bayesian_network import BayesianNetwork
from structure_learning import learn_structure
from inference import Inference

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class BayesianModel:
    def __init__(self, method='k2', max_parents=2, iterations=100, categorical_columns=None):
        self.method = method
        self.max_parents = max_parents
        self.iterations = iterations
        self.categorical_columns = categorical_columns or []
        self.network = BayesianNetwork(method=method, max_parents=max_parents, iterations=iterations, categorical_columns=categorical_columns)
        self.data = None

    def fit(self, data: pd.DataFrame, prior_edges: List[Tuple[str, str]] = None, progress_callback=None):
        self.data = data
        self.network.fit(data, prior_edges, progress_callback)
        if progress_callback:
            progress_callback("Model fitting complete")
        logger.info("Bayesian model fitting complete")

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for the Bayesian network."""
        return self.network.preprocess_data(data)

    def evaluate(self, data: pd.DataFrame, num_folds: int = 5) -> float:
        """Evaluate the Bayesian network using cross-validation."""
        return self.cross_validate(data, num_folds)

    def compute_sensitivity(self, target_node: str, num_samples: int = 1000) -> Dict[str, float]:
        """Compute sensitivity of the target node to changes in other nodes."""
        return self.network.compute_sensitivity(target_node, num_samples)

    def analyze_network(self):
        analysis = {}
        
        # Compute marginal likelihoods
        analysis['marginal_likelihoods'] = self.network.compute_marginal_likelihoods()
        
        # Compute posterior probabilities of edges
        analysis['edge_probabilities'] = self.network.compute_edge_probabilities()
        
        # Identify most influential nodes
        analysis['influential_nodes'] = self.network.identify_influential_nodes()
        
        return analysis

    def log_likelihood(self, data: pd.DataFrame) -> float:
        return self.network._compute_log_likelihood(data)

    @property
    def nodes(self):
        return self.network.nodes

    def save(self, filename: str):
        """Save the Bayesian network to a file."""
        self.network.save(filename)

    def write_results_to_json(self, results):
        self.write_results_to_json(results)

    def explain_structure_extended(self):
        return self.network.explain_structure_extended()

    @classmethod
    def load(cls, filename: str):
        """Load the Bayesian network from a file."""
        network = BayesianNetwork.load(filename)
        model = cls()
        model.network = network
        return model


    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed_data = data.copy()
        for node_name, node in self.nodes.items():
            if node.parents:
                parent_values = transformed_data[[parent.name for parent in node.parents]]
                transformed_data[node_name] = node.predict(parent_values)
        return transformed_data
    
    def set_parameters(self, node_name, values, parent_variables):
        self.nodes[node_name].set_parameters(values, parent_variables)

    def _fit_parameters(self, data):
        for node_name, node in self.nodes.items():
            parent_names = [parent.name for parent in node.parents]
            node_data = data[node_name]
            parent_data = data[parent_names] if parent_names else None

            try:
                if isinstance(node, CategoricalNode):
                    node.fit(node_data, parent_data)
                else:
                    if parent_data is not None:
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression().fit(parent_data, node_data)
                        node.set_distribution((model.intercept_, model.coef_))
                    else:
                        node.set_distribution((node_data.mean(), node_data.std()))
                
                node.fitted = True
                
            except Exception as e:
                print(f"Error fitting node {node_name}: {str(e)}")
                raise

    def topological_sort(self) -> List[str]:
        graph = nx.DiGraph(self.network.edges)
        return list(nx.topological_sort(graph))

    def cross_validate(self, data: pd.DataFrame, num_folds: int = 5) -> float:
        fold_size = len(data) // num_folds
        log_predictive_densities = []

        for i in range(num_folds):
            test_data = data.iloc[i*fold_size:(i+1)*fold_size]
            train_data = pd.concat([data.iloc[:i*fold_size], data.iloc[(i+1)*fold_size:]])
            
            # Fit the model on training data
            self.fit(train_data)
            
            # Compute log predictive density for test data
            log_pred_density = self.compute_log_predictive_density(test_data)
            log_predictive_densities.append(log_pred_density)
        
        # Compute and return the log predictive density
        return logsumexp(log_predictive_densities) - np.log(num_folds)

    def compute_log_predictive_density(self, data: pd.DataFrame) -> float:
        log_density = 0
        for _, row in data.iterrows():
            log_density += self.log_likelihood(row)
        return log_density
    
    def get_intervention_simulations(self, num_simulations=1000):
        interventions = {
            'increase_gray_matter': {'FS_Total_GM_Vol': lambda x: x * 1.1},
            'increase_white_matter': {'FS_Tot_WM_Vol': lambda x: x * 1.1},
            'increase_openness': {'NEOFAC_O': lambda x: min(x + 1, 5)},  # Assuming NEOFAC_O is on a 1-5 scale
        }

        results = {}
        for intervention_name, intervention in interventions.items():
            simulated_data = self.simulate_intervention(intervention, size=num_simulations)
            results[intervention_name] = {
                'CogFluidComp_Unadj': simulated_data['CogFluidComp_Unadj'].mean(),
                'CogCrystalComp_Unadj': simulated_data['CogCrystalComp_Unadj'].mean(),
            }

        return results

    def simulate_intervention(self, intervention, size=1000):
        if self.data is None:
            raise ValueError("No data available. Please fit the model first.")
        
        simulated_data = self.data.sample(n=size, replace=True).copy()
        for var, func in intervention.items():
            simulated_data[var] = simulated_data[var].apply(func)

        # Propagate the effects through the network
        sorted_nodes = self.topological_sort()
        for node in sorted_nodes:
            if node not in intervention:
                parents = self.network.get_parents(node)
                if parents:
                    parent_values = simulated_data[parents]
                    simulated_data[node] = self.predict_node(node, parent_values)

        return simulated_data

    def predict_node(self, node, parent_values):
        if node not in self.network.nodes:
            raise ValueError(f"Node {node} not in network")
        
        if not self.network.get_parents(node):
            return np.full(len(parent_values), self.network.parameters[node]['mean'])
        else:
            return parent_values @ self.network.parameters[node]['beta']
    