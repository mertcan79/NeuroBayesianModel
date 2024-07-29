import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import networkx as nx
import random
from bayesian_node import BayesianNode
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency

logger = logging.getLogger(__name__)

def learn_structure(
    data: pd.DataFrame,
    method: str = "nsl",
    max_parents: int = 6,
    iterations: int = 1500,
    prior_edges: Optional[List[Tuple[str, str]]] = None,
    alpha: float = 0.05
) -> List[Tuple[str, str]]:
    """Main function to learn the structure of the Bayesian network."""
    if method == "nsl":
        return neurological_structure_learning(
            data=data, 
            max_parents=max_parents, 
            iterations=iterations, 
            prior_edges=prior_edges
        )
    else:
        raise ValueError(f"Unsupported structure learning method: {method}")

def neurological_structure_learning(
    data: pd.DataFrame,
    max_parents: int = 6,
    iterations: int = 1500,
    prior_edges: Optional[List[Tuple[str, str]]] = None
) -> List[Tuple[str, str]]:
    """Learn the structure of the Bayesian network using a MCMC approach."""
    # Initialize the graph
    G = nx.DiGraph()
    G.add_nodes_from(data.columns)
    
    # Validate that prior_edges is a list of tuples
    if prior_edges is not None:
        assert isinstance(prior_edges, list), "prior_edges should be a list"
        assert all(isinstance(edge, tuple) and len(edge) == 2 for edge in prior_edges), "All elements in prior_edges should be tuples of length 2"
        G.add_edges_from(prior_edges)
    
    def score_function(data, parents, node):
        if not parents:
            return -np.inf

        parents_data = data[parents].to_numpy()
        node_data = data[node].to_numpy()

        if parents_data.shape[0] != node_data.shape[0]:
            raise ValueError("Inconsistent number of samples between parents and node data.")

        parents_data_agg = np.sum(parents_data, axis=1)

        if np.any(np.isnan(parents_data_agg)) or np.any(np.isnan(node_data)):
            return -np.inf

        if np.issubdtype(parents_data_agg.dtype, np.number) and np.issubdtype(node_data.dtype, np.number):
            parents_data_agg = np.digitize(parents_data_agg, bins=np.linspace(min(parents_data_agg), max(parents_data_agg), num=10))
            node_data = np.digitize(node_data, bins=np.linspace(min(node_data), max(node_data), num=10))

        return mutual_info_score(parents_data_agg, node_data)

    def get_neighborhood(G: nx.DiGraph) -> List[Tuple[str, str]]:
        nodes = list(G.nodes())
        return [(i, j) for i in nodes for j in nodes if i != j]

    current_score = sum(score_function(data, list(G.predecessors(node)), node) for node in G.nodes())
    
    for _ in range(iterations):
        i, j = random.choice(get_neighborhood(G))
        
        G_new = G.copy()
        if G_new.has_edge(i, j):
            G_new.remove_edge(i, j)
        else:
            G_new.add_edge(i, j)
            
        if nx.is_directed_acyclic_graph(G_new):
            new_score = sum(score_function(data, list(G_new.predecessors(node)), node) for node in G_new.nodes())
            
            if np.isnan(new_score) or np.isinf(new_score):
                continue

            if new_score > current_score or random.random() < np.exp(new_score - current_score):
                G = G_new
                current_score = new_score

    dynamic_edges = []
    for edge in G.edges():
        source, target = edge
        correlation = data[source].rolling(window=20).corr(data[target])
        if correlation.std() > 0.1:
            dynamic_edges.append(edge)
    
    nx.set_edge_attributes(G, {edge: {'dynamic': True} for edge in dynamic_edges})
    
    learned_edges = list(G.edges())
    
    # Add dynamic edge information
    dynamic_edges = []
    for edge in learned_edges:
        source, target = edge
        correlation = data[source].rolling(window=20).corr(data[target])
        if correlation.std() > 0.1:
            dynamic_edges.append(edge)
    
    # Return the list of learned edges
    return learned_edges

def k2_algorithm(data: pd.DataFrame, max_parents: int, prior_edges: Optional[List[Tuple[str, str]]] = None) -> Dict[str, BayesianNode]:
    """K2 algorithm for structure learning."""
    nodes = {column: BayesianNode(column) for column in data.columns}
    
    for node_name in data.columns:
        best_parents = []
        best_score = score_node(data, node_name, [])
        
        while len(best_parents) < max_parents:
            new_parent = None
            new_score = best_score
            
            for potential_parent in data.columns:
                if potential_parent != node_name and potential_parent not in best_parents:
                    score = score_node(data, node_name, best_parents + [potential_parent])
                    if score > new_score:
                        new_parent = potential_parent
                        new_score = score
            
            if new_parent is None:
                break
            
            best_parents.append(new_parent)
            best_score = new_score
        
        nodes[node_name].parents = [nodes[parent] for parent in best_parents]
    
    return nodes

def score_node(
    data: pd.DataFrame,
    node: str,
    parents: List[str],
    prior_edges: Optional[Dict[Tuple[str, str], float]] = None,
    allowed_connections: Optional[List[Tuple[str, str]]] = None
) -> float:
    """Compute the score for a node given its parents."""
    if not parents:
        return -np.inf

    X = data[parents]
    y = data[node]

    if allowed_connections:
        if not all((parent, node) in allowed_connections for parent in parents):
            return -np.inf

    prior_score = (
        sum(prior_edges.get((parent, node), 0) for parent in parents)
        if prior_edges
        else 0
    )

    residuals = y - X.mean()
    likelihood_score = -np.sum(residuals**2)

    complexity_penalty = -0.5 * len(parents) * np.log(len(data))

    return likelihood_score + complexity_penalty + prior_score