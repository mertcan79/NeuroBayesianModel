import json
import os
from typing import Dict, Any, Tuple, Callable, List
from datetime import datetime
from functools import lru_cache
from joblib import Parallel, delayed
import logging

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, multivariate_normal
import networkx as nx
import statsmodels.api as sm

from .bayesian_node import BayesianNode, CategoricalNode
from .structure_learning import learn_structure
from .parameter_fitting import fit_parameters


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianNetwork:
    def __init__(self, method="k2", max_parents=2, iterations=300, categorical_columns=None):
        self.method = method
        self.max_parents = max_parents
        self.iterations = iterations
        self.categorical_columns = categorical_columns or []
        self.nodes = {}
        self.edges = []
        self.data = None

    def fit(self, data: pd.DataFrame, prior: Dict[str, Any] = None, max_iter: int = 100, tol: float = 1e-6):
        self.data = data
        self._initialize_parameters(data, prior)
        
        log_likelihood_old = -np.inf
        for iteration in range(max_iter):
            responsibilities = self._expectation_step(data)
            self._maximization_step(data, responsibilities)
            log_likelihood_new = self._compute_log_likelihood(data)
            
            if abs(log_likelihood_new - log_likelihood_old) < tol:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            log_likelihood_old = log_likelihood_new
        
        if iteration == max_iter - 1:
            print(f"Did not converge after {max_iter} iterations")

    def _initialize_parameters(self, data: pd.DataFrame, prior: Dict[str, Any]):
        for node in self.nodes:
            parents = [edge[0] for edge in self.edges if edge[1] == node]
            if not parents:
                self.parameters[node] = {
                    'mean': data[node].mean(),
                    'std': data[node].std()
                }
            else:
                X = data[parents]
                y = data[node]
                beta = np.linalg.inv(X.T @ X) @ X.T @ y
                residuals = y - X @ beta
                self.parameters[node] = {
                    'beta': beta,
                    'std': residuals.std()
                }

    def _expectation_step(self, data: pd.DataFrame):
        responsibilities = np.zeros((len(data), len(self.nodes)))
        for i, node in enumerate(self.nodes):
            parents = [edge[0] for edge in self.edges if edge[1] == node]
            if not parents:
                responsibilities[:, i] = norm.pdf(data[node], 
                                                  loc=self.parameters[node]['mean'], 
                                                  scale=self.parameters[node]['std'])
            else:
                X = data[parents]
                y = data[node]
                mean = X @ self.parameters[node]['beta']
                responsibilities[:, i] = norm.pdf(y, loc=mean, scale=self.parameters[node]['std'])
        
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def _maximization_step(self, data: pd.DataFrame, responsibilities):
        for i, node in enumerate(self.nodes):
            parents = [edge[0] for edge in self.edges if edge[1] == node]
            weighted_data = data[node] * responsibilities[:, i]
            
            if not parents:
                self.parameters[node]['mean'] = weighted_data.sum() / responsibilities[:, i].sum()
                self.parameters[node]['std'] = np.sqrt(
                    ((data[node] - self.parameters[node]['mean'])**2 * responsibilities[:, i]).sum() 
                    / responsibilities[:, i].sum()
                )
            else:
                X = data[parents]
                y = data[node]
                weighted_X = X * responsibilities[:, i][:, np.newaxis]
                weighted_y = y * responsibilities[:, i]
                self.parameters[node]['beta'] = np.linalg.solve(weighted_X.T @ X, weighted_X.T @ y)
                residuals = y - X @ self.parameters[node]['beta']
                self.parameters[node]['std'] = np.sqrt(
                    (residuals**2 * responsibilities[:, i]).sum() / responsibilities[:, i].sum()
                )

    def _compute_log_likelihood(self, data: pd.DataFrame):
        log_likelihood = 0
        for node in self.nodes:
            parents = [edge[0] for edge in self.edges if edge[1] == node]
            if not parents:
                log_likelihood += norm.logpdf(data[node], 
                                              loc=self.parameters[node]['mean'], 
                                              scale=self.parameters[node]['std']).sum()
            else:
                X = data[parents]
                y = data[node]
                mean = X @ self.parameters[node]['beta']
                log_likelihood += norm.logpdf(y, loc=mean, scale=self.parameters[node]['std']).sum()
        return log_likelihood

    def predict(self, new_data: pd.DataFrame):
        predictions = {}
        for node in self.nodes:
            parents = [edge[0] for edge in self.edges if edge[1] == node]
            if not parents:
                predictions[node] = np.full(len(new_data), self.parameters[node]['mean'])
            else:
                X = new_data[parents]
                predictions[node] = X @ self.parameters[node]['beta']
        return pd.DataFrame(predictions)

    def get_edges(self):
        return self.edges

    def explain_structure(self):
        return {"nodes": list(self.nodes.keys()), "edges": self.get_edges()}

    def explain_structure_extended(self):
        structure = self.explain_structure()
        for node_name, node in self.nodes.items():
            structure[node_name] = {
                "parents": [edge[0] for edge in self.edges if edge[1] == node_name],
                "children": [edge[1] for edge in self.edges if edge[0] == node_name],
                "parameters": self.parameters[node_name] if node_name in self.parameters else None,
            }
        return structure

    def topological_sort(self) -> List[str]:
        graph = nx.DiGraph(self.edges)
        return list(nx.topological_sort(graph))
    @lru_cache(maxsize=128)
    def sample_node(self, node_name: str, size: int = 1) -> np.ndarray:
        sorted_nodes = self.topological_sort()
        samples = {node: None for node in sorted_nodes}

        for node in sorted_nodes:
            if node == node_name:
                break
            parent_values = {
                parent: samples[parent] for parent in self.nodes[node].parents
            }
            samples[node] = self.nodes[node].sample(size, parent_values)

        return samples[node_name]

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.apply(
            lambda col: (
                pd.Categorical(col).codes
                if col.name in self.categorical_columns
                else col
            )
        )

    def _learn_structure(self, data: pd.DataFrame):
        self.nodes = learn_structure(
            data,
            method=self.method,
            max_parents=self.max_parents,
            iterations=self.iterations,
            prior_edges=self.prior_edges,
        )

        # Enforce max_parents limit
        for node_name, node in self.nodes.items():
            if len(node.parents) > self.max_parents:
                print(
                    f"Warning: Node {node_name} has {len(node.parents)} parents. Limiting to {self.max_parents}."
                )
                node.parents = node.parents[: self.max_parents]

    def _create_nodes(self, data: pd.DataFrame):
        for column in data.columns:
            if column in self.categorical_columns:
                categories = data[column].astype("category").cat.categories.tolist()
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
                log_probs += np.array(
                    [
                        node.log_probability(val, tuple(pv))
                        for val, pv in zip(data[node_name], parent_values)
                    ]
                )
            else:
                log_probs += np.array(
                    [node.log_probability(val) for val in data[node_name]]
                )
        return np.sum(log_probs)

    def compute_sensitivity(
        self, target_node: str, num_samples: int = 1000
    ) -> Dict[str, float]:
        sensitivity = {}
        target = self.nodes[target_node]

        if isinstance(target.distribution, tuple):
            base_mean, base_std = target.distribution
            base_mean = (
                np.mean(base_mean) if isinstance(base_mean, np.ndarray) else base_mean
            )
            base_std = (
                np.mean(base_std) if isinstance(base_std, np.ndarray) else base_std
            )
        else:
            base_mean, base_std = np.mean(target.distribution), np.std(
                target.distribution
            )

        base_std = max(base_std, 1e-10)  # Avoid division by zero

        for node_name, node in self.nodes.items():
            if node_name != target_node:
                if isinstance(node.distribution, tuple):
                    mean, std = node.distribution
                    mean = np.mean(mean) if isinstance(mean, np.ndarray) else mean
                    std = np.mean(std) if isinstance(std, np.ndarray) else std
                else:
                    mean, std = np.mean(node.distribution), np.std(node.distribution)

                std = max(std, 1e-10)  # Avoid using zero standard deviation

                # Calculate relative change
                delta_input = std / mean if np.abs(mean) > 1e-10 else 1

                if node in target.parents:
                    if isinstance(target.distribution, tuple):
                        coef = target.distribution[1][target.parents.index(node)]
                        coef = np.mean(coef) if isinstance(coef, np.ndarray) else coef
                    else:
                        coef = 1
                    delta_output = coef * delta_input
                else:
                    delta_output = 0.1 * delta_input  # Assume small indirect effect

                # Use relative change in sensitivity calculation
                sensitivity[node_name] = (
                    (delta_output / base_mean) / (delta_input / mean)
                    if np.abs(mean) > 1e-10
                    else 0
                )

        return sensitivity

    def compute_edge_strength(self, parent_name: str, child_name: str) -> float:
        parent_node = self.nodes[parent_name]
        child_node = self.nodes[child_name]

        if (
            isinstance(child_node.distribution, tuple)
            and len(child_node.distribution) == 2
        ):
            coefficients = child_node.distribution[1]
            parent_index = child_node.parents.index(parent_node)
            return abs(coefficients[parent_index])
        else:
            # If the distribution is not in the expected format, return a default value
            return 0.0

    def get_performance_metrics(self):
        # Assuming we've already performed cross-validation
        mse = np.mean((self.y_true - self.y_pred) ** 2)
        r2 = 1 - (
            np.sum((self.y_true - self.y_pred) ** 2)
            / np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        )

        return {
            "Mean Squared Error": mse,
            "R-squared": r2,
            "Accuracy (within 10% of true value)": np.mean(
                np.abs(self.y_true - self.y_pred) / self.y_true < 0.1
            ),
        }

    def _expectation_step(self, data: pd.DataFrame):
        responsibilities = {}
        for node in self.nodes:
            parents = [edge[0] for edge in self.edges if edge[1] == node]
            if not parents:
                # For nodes without parents, responsibility is 1
                responsibilities[node] = np.ones(len(data))
            else:
                X = data[parents].values
                y = data[node].values
                beta = self.parameters[node]['beta']
                std = self.parameters[node]['std']
                
                mean = X @ beta
                prob = multivariate_normal.pdf(y, mean=mean, cov=std**2)
                responsibilities[node] = prob / prob.sum()
        
        return responsibilities

    def _maximization_step(self, data: pd.DataFrame, responsibilities):
        for node in self.nodes:
            parents = [edge[0] for edge in self.edges if edge[1] == node]
            if not parents:
                weighted_sum = (responsibilities[node] * data[node]).sum()
                total_weight = responsibilities[node].sum()
                self.parameters[node]['mean'] = weighted_sum / total_weight
                self.parameters[node]['std'] = np.sqrt(
                    ((data[node] - self.parameters[node]['mean'])**2 * responsibilities[node]).sum() / total_weight
                )
            else:
                X = data[parents].values
                y = data[node].values
                weights = responsibilities[node]
                
                # Weighted least squares
                weighted_X = X * np.sqrt(weights)[:, np.newaxis]
                weighted_y = y * np.sqrt(weights)
                beta = np.linalg.inv(weighted_X.T @ weighted_X) @ weighted_X.T @ weighted_y
                
                residuals = y - X @ beta
                self.parameters[node]['beta'] = beta
                self.parameters[node]['std'] = np.sqrt(
                    (weights * residuals**2).sum() / weights.sum()
                )

    def get_posterior_predictive(self, new_data):
        # This is a placeholder - the actual implementation would depend on your specific model
        predictions = {}
        for node in self.nodes:
            if not self.nodes[node].parents:
                predictions[node] = self.nodes[node].sample(len(new_data))
            else:
                parent_values = {
                    parent.name: predictions[parent.name]
                    for parent in self.nodes[node].parents
                }
                predictions[node] = self.nodes[node].sample(
                    len(new_data), parent_values
                )
        return predictions


    def get_confidence_intervals(self):
        ci_results = {}
        for node_name, node in self.nodes.items():
            if node.parents:
                X = self.data[[p.name for p in node.parents]]
                y = self.data[node_name]

                # Add a constant term to the predictors
                X = sm.add_constant(X)

                # Perform multiple linear regression
                model = sm.OLS(y, X).fit()

                ci_results[node_name] = {
                    "coefficients": model.params[1:].to_dict(),  # Exclude the intercept
                    "intercept": model.params.iloc[0],  # Get the intercept
                    "ci_lower": model.conf_int()[0].to_dict(),
                    "ci_upper": model.conf_int()[1].to_dict(),
                    "p_values": model.pvalues.to_dict(),
                }
        return ci_results

    def fit_transform(
        self,
        data: pd.DataFrame,
        prior_edges: List[tuple] = None,
        progress_callback: Callable[[float], None] = None,
    ):
        self.fit(data, prior_edges=prior_edges, progress_callback=progress_callback)
        return self.transform(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed_data = data.copy()
        for col in self.categorical_columns:
            if col in transformed_data:
                transformed_data[col] = (
                    transformed_data[col].astype("category").cat.codes
                )
        return transformed_data

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def fit(self, data: pd.DataFrame, prior: Dict[str, Any] = None, max_iter: int = 100, tol: float = 1e-6):
        """
        Fit the Bayesian Network using Expectation-Maximization (EM) algorithm.
        
        :param data: DataFrame containing the data
        :param prior: Prior distributions for parameters
        :param max_iter: Maximum number of iterations for EM
        :param tol: Convergence tolerance
        """
        # Initialize parameters
        self._initialize_parameters(data, prior)
        
        log_likelihood_old = -np.inf
        for iteration in range(max_iter):
            # E-step
            responsibilities = self._expectation_step(data)
            
            # M-step
            self._maximization_step(data, responsibilities)
            
            # Compute log-likelihood
            log_likelihood_new = self._compute_log_likelihood(data)
            
            # Check for convergence
            if abs(log_likelihood_new - log_likelihood_old) < tol:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            log_likelihood_old = log_likelihood_new
        
        if iteration == max_iter - 1:
            print(f"Did not converge after {max_iter} iterations")

    def _initialize_parameters(self, data: pd.DataFrame, prior: Dict[str, Any]):
        for node in self.nodes:
            parents = [edge[0] for edge in self.edges if edge[1] == node]
            if not parents:
                self.parameters[node] = {
                    'mean': data[node].mean(),
                    'std': data[node].std()
                }
            else:
                X = data[parents]
                y = data[node]
                beta = np.linalg.inv(X.T @ X) @ X.T @ y
                residuals = y - X @ beta
                self.parameters[node] = {
                    'beta': beta,
                    'std': residuals.std()
                }

    def _compute_log_likelihood(self, data: pd.DataFrame):
        log_likelihood = 0
        for node in self.nodes:
            parents = [edge[0] for edge in self.edges if edge[1] == node]
            if not parents:
                log_likelihood += norm.logpdf(data[node], 
                                              loc=self.parameters[node]['mean'], 
                                              scale=self.parameters[node]['std']).sum()
            else:
                X = data[parents]
                y = data[node]
                mean = X @ self.parameters[node]['beta']
                log_likelihood += norm.logpdf(y, loc=mean, scale=self.parameters[node]['std']).sum()
        return log_likelihood

    def _expectation_step(self, data: pd.DataFrame):
        responsibilities = np.zeros((len(data), len(self.nodes)))
        for i, node in enumerate(self.nodes):
            parents = [edge[0] for edge in self.edges if edge[1] == node]
            if not parents:
                responsibilities[:, i] = norm.pdf(data[node], 
                                                loc=self.parameters[node]['mean'], 
                                                scale=self.parameters[node]['std'])
            else:
                X = data[parents]
                y = data[node]
                mean = X @ self.parameters[node]['beta']
                responsibilities[:, i] = norm.pdf(y, loc=mean, scale=self.parameters[node]['std'])
        
        # Normalize responsibilities
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def _maximization_step(self, data: pd.DataFrame, responsibilities):
        for i, node in enumerate(self.nodes):
            parents = [edge[0] for edge in self.edges if edge[1] == node]
            weighted_data = data[node] * responsibilities[:, i]
            
            if not parents:
                self.parameters[node]['mean'] = weighted_data.sum() / responsibilities[:, i].sum()
                self.parameters[node]['std'] = np.sqrt(
                    ((data[node] - self.parameters[node]['mean'])**2 * responsibilities[:, i]).sum() 
                    / responsibilities[:, i].sum()
                )
            else:
                X = data[parents]
                y = data[node]
                weighted_X = X * responsibilities[:, i][:, np.newaxis]
                weighted_y = y * responsibilities[:, i]
                self.parameters[node]['beta'] = np.linalg.solve(weighted_X.T @ X, weighted_X.T @ y)
                residuals = y - X @ self.parameters[node]['beta']
                self.parameters[node]['std'] = np.sqrt(
                    (residuals**2 * responsibilities[:, i]).sum() / responsibilities[:, i].sum()
                )

    def compute_node_marginal_likelihood(self, node):
        if node not in self.nodes:
            raise ValueError(f"Node {node} not in network")
        
        node_data = self.data[node]
        parents = self.get_parents(node)
        
        if not parents:
            # For nodes without parents, use a simple Gaussian likelihood
            mean = np.mean(node_data)
            std = np.std(node_data)
            log_likelihood = np.sum(norm.logpdf(node_data, mean, std))
        else:
            # For nodes with parents, use linear regression
            parent_data = self.data[parents]
            X = sm.add_constant(parent_data)
            y = node_data
            model = sm.OLS(y, X).fit()
            log_likelihood = -0.5 * model.nobs * np.log(2 * np.pi) - 0.5 * model.nobs * np.log(model.scale) - 0.5 * model.nobs

        # Add log prior (assuming uniform prior over structures)
        log_prior = 0
        
        return log_likelihood + log_prior

    def compute_edge_probability(self, edge):
        parent, child = edge
        if parent not in self.nodes or child not in self.nodes:
            raise ValueError(f"Edge {edge} not in network")
        
        # Compute marginal likelihood with and without the edge
        log_ml_with_edge = self.compute_node_marginal_likelihood(child)
        
        # Temporarily remove the edge
        self.remove_edge(parent, child)
        log_ml_without_edge = self.compute_node_marginal_likelihood(child)
        self.add_edge(parent, child)  # Add the edge back
        
        # Compute edge probability using Bayes factor
        log_bayes_factor = log_ml_with_edge - log_ml_without_edge
        edge_prob = 1 / (1 + np.exp(-log_bayes_factor))
        
        return edge_prob

    def compute_node_influence(self, node):
        if node not in self.nodes:
            raise ValueError(f"Node {node} not in network")
        
        # Compute total effect on children
        total_effect = 0
        for child in self.get_children(node):
            edge_prob = self.compute_edge_probability((node, child))
            child_data = self.data[child]
            node_data = self.data[node]
            correlation = np.corrcoef(node_data, child_data)[0, 1]
            total_effect += edge_prob * abs(correlation)
        
        return total_effect

    def compute_pairwise_mutual_information(self, node1, node2):
        if node1 not in self.nodes or node2 not in self.nodes:
            raise ValueError(f"Nodes {node1} or {node2} not in network")
        
        data1 = self.data[node1]
        data2 = self.data[node2]
        
        # Compute joint probability
        hist_2d, _, _ = np.histogram2d(data1, data2, bins=20)
        p_xy = hist_2d / float(np.sum(hist_2d))
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)
        
        # Compute mutual information
        mi = np.sum(p_xy * np.log(p_xy / (p_x[:, None] * p_y[None, :])))
        
        return mi

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

    def fit(self, data: pd.DataFrame, prior: Dict[str, Any] = None, max_iter: int = 100, tol: float = 1e-6):
        """
        Fit the Bayesian Network using Expectation-Maximization (EM) algorithm.
        
        :param data: DataFrame containing the data
        :param prior: Prior distributions for parameters
        :param max_iter: Maximum number of iterations for EM
        :param tol: Convergence tolerance
        """
        # Initialize parameters
        self._initialize_parameters(data, prior)
        
        log_likelihood_old = -np.inf
        for iteration in range(max_iter):
            # E-step
            responsibilities = self._expectation_step(data)
            
            # M-step
            self._maximization_step(data, responsibilities)
            
            # Compute log-likelihood
            log_likelihood_new = self._compute_log_likelihood(data)
            
            # Check for convergence
            if abs(log_likelihood_new - log_likelihood_old) < tol:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            log_likelihood_old = log_likelihood_new
        
        if iteration == max_iter - 1:
            print(f"Did not converge after {max_iter} iterations")

    def _initialize_parameters(self, data: pd.DataFrame, prior: Dict[str, Any]):
        for node in self.nodes:
            parents = [edge[0] for edge in self.edges if edge[1] == node]
            if not parents:
                self.parameters[node] = {
                    'mean': data[node].mean(),
                    'std': data[node].std()
                }
            else:
                X = data[parents]
                y = data[node]
                beta = np.linalg.inv(X.T @ X) @ X.T @ y
                residuals = y - X @ beta
                self.parameters[node] = {
                    'beta': beta,
                    'std': residuals.std()
                }

    def _compute_log_likelihood(self, data: pd.DataFrame):
        log_likelihood = 0
        for node in self.nodes:
            parents = [edge[0] for edge in self.edges if edge[1] == node]
            if not parents:
                log_likelihood += norm.logpdf(data[node], 
                                              loc=self.parameters[node]['mean'], 
                                              scale=self.parameters[node]['std']).sum()
            else:
                X = data[parents]
                y = data[node]
                mean = X @ self.parameters[node]['beta']
                log_likelihood += norm.logpdf(y, loc=mean, scale=self.parameters[node]['std']).sum()
        return log_likelihood
