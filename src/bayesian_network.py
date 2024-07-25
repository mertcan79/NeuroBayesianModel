import json
import os
from typing import Dict, Any, Tuple, Callable, List
from datetime import datetime
from functools import lru_cache
import logging

import pandas as pd
import numpy as np
from scipy.stats import norm, multivariate_normal
import networkx as nx
import statsmodels.api as sm

from bayesian_node import BayesianNode, CategoricalNode
from structure_learning import learn_structure
from parameter_fitting import fit_parameters

logging.basicConfig(level=logging.ERROR)
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
        self.parameters = {}

    def fit(self, data: pd.DataFrame, prior: Dict[str, Any] = None, max_iter: int = 100, tol: float = 1e-6):
        self.data = data
        self._learn_structure(data)
        self._initialize_parameters(data, prior)
        
        log_likelihood_old = -np.inf
        for iteration in range(max_iter):
            responsibilities = self._expectation_step(data)
            self._maximization_step(data, responsibilities)
            log_likelihood_new = self._compute_log_likelihood(data)
            
            if abs(log_likelihood_new - log_likelihood_old) < tol:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
            
            log_likelihood_old = log_likelihood_new
        
        if iteration == max_iter - 1:
            logger.warning(f"Did not converge after {max_iter} iterations")

    def _learn_structure(self, data: pd.DataFrame):
        self.nodes = learn_structure(
            data,
            method=self.method,
            max_parents=self.max_parents,
            iterations=self.iterations
        )
        self.edges = [(parent.name, node.name) for node in self.nodes.values() for parent in node.parents]

    def _initialize_parameters(self, data: pd.DataFrame, prior: Dict[str, Any]):
        for node in self.nodes:
            parents = self.get_parents(node)
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
        responsibilities = {}
        for node in self.nodes:
            parents = self.get_parents(node)
            if not parents:
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
            parents = self.get_parents(node)
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
                
                weighted_X = X * np.sqrt(weights)[:, np.newaxis]
                weighted_y = y * np.sqrt(weights)
                beta = np.linalg.inv(weighted_X.T @ weighted_X) @ weighted_X.T @ weighted_y
                
                residuals = y - X @ beta
                self.parameters[node]['beta'] = beta
                self.parameters[node]['std'] = np.sqrt(
                    (weights * residuals**2).sum() / weights.sum()
                )

    def _compute_log_likelihood(self, data: pd.DataFrame):
        log_likelihood = 0
        for node in self.nodes:
            parents = self.get_parents(node)
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
            parents = self.get_parents(node)
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
        for node_name in self.nodes:
            structure[node_name] = {
                "parents": self.get_parents(node_name),
                "children": self.get_children(node_name),
                "parameters": self.parameters[node_name] if node_name in self.parameters else None,
            }
        return structure

    def topological_sort(self) -> List[str]:
        graph = nx.DiGraph(self.edges)
        return list(nx.topological_sort(graph))

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        for col in self.categorical_columns:
            if col in data.columns:
                data[col] = pd.Categorical(data[col]).codes
        return data

    def add_edge(self, parent, child):
        if parent not in self.nodes or child not in self.nodes:
            raise ValueError("Both nodes must exist in the network")
        self.edges.append((parent, child))
        self.nodes[child].parents.append(self.nodes[parent])
        self.nodes[parent].children.append(self.nodes[child])

    def remove_edge(self, parent, child):
        self.edges.remove((parent, child))
        self.nodes[child].parents.remove(self.nodes[parent])
        self.nodes[parent].children.remove(self.nodes[child])

    def get_parents(self, node: str) -> List[str]:
        return [edge[0] for edge in self.edges if edge[1] == node]

    def get_children(self, node: str) -> List[str]:
        return [edge[1] for edge in self.edges if edge[0] == node]

    @lru_cache(maxsize=128)
    def sample_node(self, node_name: str, size: int = 1) -> np.ndarray:
        sorted_nodes = self.topological_sort()
        samples = {node: None for node in sorted_nodes}

        for node in sorted_nodes:
            if node == node_name:
                break
            parent_values = {parent: samples[parent] for parent in self.get_parents(node)}
            samples[node] = self.nodes[node].sample(size, parent_values)

        return samples[node_name]

    def get_confidence_intervals(self):
        ci_results = {}
        for node_name in self.nodes:
            parents = self.get_parents(node_name)
            if parents:
                X = self.data[parents]
                y = self.data[node_name]

                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()

                ci_results[node_name] = {
                    "coefficients": model.params[1:].to_dict(),
                    "intercept": model.params.iloc[0],
                    "ci_lower": model.conf_int()[0].to_dict(),
                    "ci_upper": model.conf_int()[1].to_dict(),
                    "p_values": model.pvalues.to_dict(),
                }
        return ci_results

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self):
        return {
            'nodes': {name: node.to_dict() for name, node in self.nodes.items()},
            'edges': self.edges,
            'method': self.method,
            'max_parents': self.max_parents,
            'iterations': self.iterations,
            'categorical_columns': self.categorical_columns,
            'parameters': self.parameters
        }

    @classmethod
    def from_dict(cls, data):
        network = cls(
            method=data['method'],
            max_parents=data['max_parents'],
            iterations=data['iterations'],
            categorical_columns=data['categorical_columns']
        )
        network.nodes = {name: BayesianNode.from_dict(node_data) for name, node_data in data['nodes'].items()}
        network.edges = data['edges']
        network.parameters = data['parameters']
        return network

    def compute_sensitivity(self, target_node: str, num_samples: int = 1000) -> Dict[str, float]:
        sensitivity = {}
        target = self.nodes[target_node]
        baseline = self.sample_node(target_node, num_samples)

        for node_name in self.nodes:
            if node_name != target_node:
                perturbed_data = self.data.copy()
                perturbed_data[node_name] += np.random.normal(0, 0.1 * perturbed_data[node_name].std(), len(perturbed_data))
                self.fit(perturbed_data)
                perturbed_samples = self.sample_node(target_node, num_samples)
                sensitivity[node_name] = np.mean(np.abs(perturbed_samples - baseline)) / np.std(baseline)

        return sensitivity

    def get_key_relationships(self) -> List[Dict[str, Any]]:
        relationships = []
        for node_name, node in self.nodes.items():
            for parent in self.get_parents(node_name):
                strength = abs(self.parameters[node_name]['beta'][self.get_parents(node_name).index(parent)])
                relationships.append({
                    "parent": parent,
                    "child": node_name,
                    "strength": strength
                })
        sorted_relationships = sorted(relationships, key=lambda x: x['strength'], reverse=True)
        top_10_percent = sorted_relationships[:max(1, len(sorted_relationships) // 10)]
        return [{"parent": r["parent"], "child": r["child"], "strength": round(r["strength"], 2)} for r in top_10_percent]

    def compute_marginal_likelihoods(self):
        marginal_likelihoods = {}
        for node in self.nodes:
            marginal_likelihoods[node] = self.compute_node_marginal_likelihood(node)
        return marginal_likelihoods

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

    def identify_influential_nodes(self):
        influential_nodes = []
        for node in self.nodes:
            influence = self.compute_node_influence(node)
            influential_nodes.append((node, influence))
        return sorted(influential_nodes, key=lambda x: x[1], reverse=True)

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

    def compute_edge_probabilities(self):
        edge_probabilities = {}
        for edge in self.edges:
            edge_probabilities[edge] = self.compute_edge_probability(edge)
        return edge_probabilities

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

