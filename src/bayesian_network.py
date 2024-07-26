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
    def __init__(self, method="k2", max_parents=2, iterations=300, categorical_columns=None):
        self.method = method
        self.max_parents = max_parents
        self.iterations = iterations
        self.categorical_columns = categorical_columns or []
        self.nodes = {}
        self.edges = []  # Change this from {} to []
        self.data = None
        self.parameters = {}
        self.inference = None

    def add_node(self, node: 'BayesianNode'):
        self.nodes[node.name] = node

    def add_edge(self, parent_name: str, child_name: str):
        if parent_name not in self.nodes or child_name not in self.nodes:
            raise ValueError("Parent or child node not found in the network")
        parent_node = self.nodes[parent_name]
        child_node = self.nodes[child_name]
        parent_node.add_child(child_node)
        child_node.add_parent(parent_node)

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

    def fit(self, data: pd.DataFrame, prior_edges=None, progress_callback=None, max_iter=100, tol=1e-3):
        self.data = data
        self.nodes = self._create_nodes_from_data(data)
        self._learn_structure(data, prior_edges)
        self._initialize_parameters(data)
        self._fit_nodes(data)
        self.inference = Inference(nodes=self.nodes)

    def _fit_nodes(self, data):
        for node_name, node in self.nodes.items():
            parent_names = self.get_parents(node_name)
            node_data = data[node_name]
            parent_data = data[parent_names] if parent_names else None
            try:
                node.fit(node_data, parent_data)
            except Exception as e:
                print(f"Error fitting node {node_name}: {str(e)}")
                # Set a default distribution for the node
                node.distribution = {'mean': np.mean(node_data), 'std': np.std(node_data) if len(node_data) > 1 else 1e-6}
                node.fitted = True

    def get_parents_for_node(self, node_name: str) -> List[str]:
        """Return a list of parent nodes for the given node."""
        parents = []
        for parent, children in self.edges.items():
            if node_name in children:
                parents.append(parent)
        return parents

    def _create_nodes_from_data(self, data: pd.DataFrame) -> Dict[str, BayesianNode]:
        # Create nodes from data
        return {column: BayesianNode(name=column) for column in data.columns}

    def sample(self, node_name: str, size: int = 1, parent_values: Optional[np.ndarray] = None) -> np.ndarray:
        node = self.get_node(node_name)
        return node.sample(size, parent_values)

    def compute_sensitivity(self, target_node_name: str, num_samples: int = 1000) -> Dict[str, float]:
        if target_node_name not in self.nodes:
            raise ValueError(f"Node {target_node_name} not found in the network.")
        
        inference = Inference(self.nodes)
        
        # Sample data for the target node
        target_samples = inference.sample_node(target_node_name, num_samples)
        
        sensitivities = {}
        for node_name, node in self.nodes.items():
            if node_name == target_node_name:
                continue
            
            # Sample data for the other node
            other_samples = inference.sample_node(node_name, num_samples)
            
            # Compute correlation as a measure of sensitivity
            sensitivity = np.corrcoef(target_samples, other_samples)[0, 1]
            sensitivities[node_name] = sensitivity
        
        return sensitivities

    def _learn_structure(self, data: pd.DataFrame, prior_edges=None):
        # Learn the structure of the Bayesian network
        self.edges = learn_structure(data, method=self.method, max_parents=self.max_parents, iterations=self.iterations, prior_edges=prior_edges)
        
        for parent, child in self.edges:
            if parent not in self.nodes or child not in self.nodes:
                print(f"Warning: Edge {parent} -> {child} references non-existent node")
                continue
            
            parent_node = self.nodes[parent]
            child_node = self.nodes[child]
            
            if child_node not in parent_node.children:
                parent_node.children.append(child_node)
            if parent_node not in child_node.parents:
                child_node.parents.append(parent_node)

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
        
        for col in self.categorical_columns:
            if col in data.columns:
                data[col] = pd.Categorical(data[col]).codes
        return data

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

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'BayesianNetwork':
        network = BayesianNetwork()
        network.nodes = data['nodes']
        network.edges = data['edges']
        network.parameters = data['parameters']
        return network
    
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

    def get_novel_insights(self):
        insights = []

        required_features = ["CogFluidComp_Unadj", "CogCrystalComp_Unadj"]
        if not all(feature in self.nodes for feature in required_features):
            return ["Required features for sensitivity analysis are missing."]

        try:
            sensitivity_fluid = self.compute_sensitivity("CogFluidComp_Unadj")
            sensitivity_crystal = self.compute_sensitivity("CogCrystalComp_Unadj")

            brain_structures = [f for f in sensitivity_fluid.keys() if f.startswith("FS_")]
            if brain_structures:
                max_influence = max(brain_structures, key=lambda x: abs(sensitivity_fluid[x]))
                insights.append(
                    f"Unexpectedly high influence of {max_influence} on fluid cognitive abilities (sensitivity: {sensitivity_fluid[max_influence]:.2f}), suggesting a potential new area for cognitive research."
                )

            if "NEOFAC_O" in sensitivity_fluid and "NEOFAC_C" in sensitivity_fluid:
                if abs(sensitivity_fluid["NEOFAC_O"]) > abs(sensitivity_fluid["NEOFAC_C"]):
                    insights.append(
                        f"Openness to experience shows a stronger relationship with fluid cognitive abilities (sensitivity: {sensitivity_fluid['NEOFAC_O']:.2f}) than conscientiousness (sensitivity: {sensitivity_fluid['NEOFAC_C']:.2f}), which could inform personality-based cognitive training approaches."
                    )

            for feature in sensitivity_fluid.keys():
                if feature in sensitivity_crystal:
                    if abs(sensitivity_fluid[feature]) > 2 * abs(sensitivity_crystal[feature]):
                        insights.append(
                            f"{feature} has a much stronger influence on fluid cognitive abilities (sensitivity: {sensitivity_fluid[feature]:.2f}) compared to crystallized abilities (sensitivity: {sensitivity_crystal[feature]:.2f}), suggesting different mechanisms for these cognitive domains."
                        )

        except Exception as e:
            insights.append(f"Error computing sensitivities: {str(e)}")

        return insights
    
    def get_clinical_implications(self):
        implications = []
        sensitivity_fluid = self.compute_sensitivity("CogFluidComp_Unadj")
        sensitivity_crystal = self.compute_sensitivity("CogCrystalComp_Unadj")

        for feature, value in sensitivity_fluid.items():
            if abs(value) > 0.1:
                implications.append(
                    f"Changes in {feature} may significantly impact fluid cognitive abilities (sensitivity: {value:.2f}), suggesting potential for targeted interventions."
                )

        for feature, value in sensitivity_crystal.items():
            if abs(value) > 0.1:
                implications.append(
                    f"Changes in {feature} may significantly impact crystallized cognitive abilities (sensitivity: {value:.2f}), indicating areas for potential cognitive preservation strategies."
                )

        if abs(sensitivity_fluid["Age"]) > 0.1 or abs(sensitivity_crystal["Age"]) > 0.1:
            implications.append(
                "Age has a substantial impact on cognitive abilities, emphasizing the need for age-specific cognitive interventions and preventive strategies."
            )

        return implications

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

