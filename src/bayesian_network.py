import json
from typing import Dict, Any, Tuple, Callable, List, Optional
from functools import lru_cache
import logging

import pandas as pd
import numpy as np
from scipy.stats import norm, multivariate_normal
import networkx as nx
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from scipy import stats

from bayesian_node import BayesianNode, CategoricalNode, Node
from structure_learning import neurological_structure_learning
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

    def get_edges(self):
        return self.edges

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

    def _learn_structure(self, data: pd.DataFrame, prior_edges: Optional[List[Tuple[str, str]]] = None):
        """Learn the structure of the Bayesian network."""
        learned_edges = neurological_structure_learning(
            data=data, 
            max_parents=self.max_parents, 
            iterations=self.iterations, 
            prior_edges=prior_edges
        )
        
        # Check if learned_edges is None or empty
        if not learned_edges:
            logger.warning("No edges were learned. The network will have no connections.")
            return

        # Check the format of learned_edges
        if not isinstance(learned_edges, (list, tuple)):
            raise ValueError(f"Expected learned_edges to be a list or tuple, but got {type(learned_edges)}")

        valid_edges = []
        for i, edge in enumerate(learned_edges):
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                logger.warning(f"Skipping invalid edge at index {i}: {edge}")
                continue
            
            parent, child = edge
            if not isinstance(parent, str) or not isinstance(child, str):
                logger.warning(f"Skipping edge with non-string nodes at index {i}: {edge}")
                continue
            
            if parent not in self.nodes or child not in self.nodes:
                logger.warning(f"Skipping edge with unknown nodes at index {i}: {edge}")
                continue
            
            valid_edges.append((parent, child))
        
        if not valid_edges:
            logger.warning("No valid edges were found. The network will have no connections.")
            return

        # Add the valid edges to the network
        for parent, child in valid_edges:
            self.add_edge(parent, child)

        logger.info(f"Added {len(valid_edges)} edges to the network.")

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

    def compute_all_edge_probabilities(self):
        edge_probabilities = {}
        for parent in self.nodes:
            for child in self.nodes:
                if parent != child:  # Avoid self-loops
                    try:
                        probability = self.compute_edge_probability(parent, child)
                        edge_probabilities[f"{parent}->{child}"] = probability
                    except ValueError:
                        continue  # Skip edges that do not exist
        return edge_probabilities

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

    def explain_structure_extended(self):
        if not hasattr(self, 'graph') or not self.graph:
            return "The network structure has not been learned yet."
        G = nx.DiGraph(self.graph)
        
        summary = []
        
        # Identify hub nodes (nodes with high degree)
        degrees = dict(G.degree())
        hub_threshold = np.percentile(list(degrees.values()), 80)  # Top 20% as hubs
        hubs = [node for node, degree in degrees.items() if degree > hub_threshold]
        
        summary.append(f"Key hub variables: {', '.join(hubs)}")
        
        # Identify strongly connected components
        components = list(nx.strongly_connected_components(G))
        if len(components) > 1:
            summary.append(f"The network has {len(components)} strongly connected components.")
        
        # Identify potential causal pathways
        cognitive_vars = ['CogFluidComp_Unadj', 'MMSE_Score']  # Example cognitive variables
        brain_vars = ['FS_L_Hippo_Vol', 'FS_R_Hippo_Vol', 'FS_Tot_WM_Vol']  # Example brain structure variables
        
        for cog_var in cognitive_vars:
            for brain_var in brain_vars:
                paths = list(nx.all_simple_paths(G, brain_var, cog_var))
                if paths:
                    summary.append(f"Potential pathway from {brain_var} to {cog_var}: {' -> '.join(paths[0])}")
        
        # Identify feedback loops
        cycles = list(nx.simple_cycles(G))
        if cycles:
            summary.append(f"The network contains {len(cycles)} feedback loops.")
            if len(cycles) <= 3:  # Only show a few examples
                summary.append(f"Example loop: {' -> '.join(cycles[0] + [cycles[0][0]])}")
        
        # Analyze edge strengths using Cramer's V for categorical variables
        edge_strengths = {}
        for parent, child in G.edges():
            if parent in self.categorical_columns or child in self.categorical_columns:
                cramer_v = self._compute_categorical_edge_strength(parent, child)
                edge_strengths[(parent, child)] = cramer_v
            else:
                correlation = self._compute_continuous_edge_strength(parent, child)
                edge_strengths[(parent, child)] = abs(correlation)
        
        # Report strongest relationships
        strongest_edges = sorted(edge_strengths.items(), key=lambda x: x[1], reverse=True)[:5]
        summary.append("Strongest relationships in the network:")
        for (parent, child), strength in strongest_edges:
            summary.append(f"  {parent} -> {child}: strength = {strength:.2f}")
        
        return "\n".join(summary)

    def _compute_categorical_edge_strength(self, parent, child):
        contingency_table = pd.crosstab(self.data[parent], self.data[child])
        chi2, _, _, _ = stats.chi2_contingency(contingency_table)
        n = self.data.shape[0]
        min_dim = min(contingency_table.shape) - 1
        cramer_v = np.sqrt(chi2 / (n * min_dim))
        return cramer_v

    def _compute_continuous_edge_strength(self, parent, child):
        correlation, _ = stats.pearsonr(self.data[parent], self.data[child])
        return correlation

    def get_key_relationships(self):
        key_relationships = {}
        for (parent, child), probability in self.compute_all_edge_probabilities().items():
            if probability > 0.5:  # Example threshold for "key" relationships
                key_relationships[(parent, child)] = probability
        return key_relationships

    def compute_sensitivity(self, target_variable: str):
        if target_variable not in self.nodes:
            raise ValueError(f"Target variable {target_variable} not found in the network.")
        
        sensitivities = {}
        for node_name in self.nodes:
            if node_name != target_variable:
                # Compute sensitivity based on mutual information
                mutual_info = self.compute_mutual_information(node_name, target_variable)
                sensitivities[node_name] = mutual_info
        
        return sensitivities

    def compute_mutual_information(self, node1: str, node2: str):
        from sklearn.metrics import mutual_info_score
        return mutual_info_score(self.data[node1], self.data[node2])