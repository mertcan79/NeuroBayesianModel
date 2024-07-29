import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
import statsmodels.api as sm
import networkx as nx
import random
from bayesian_node import BayesianNode, CategoricalNode
from sklearn.metrics import mutual_info_score

logger = logging.getLogger(__name__)

def learn_structure(data, method="nsl", max_parents=4, iterations=300, prior_edges=None, alpha=0.05):
    if method == "nsl":
        return neurological_structure_learning(data, prior_edges, max_parents, alpha)
    else:
        raise ValueError(f"Unsupported structure learning method: {method}")

def neurological_structure_learning(data, max_parents=6, iterations=1500, prior_edges=None):
    # Initialize the graph
    G = nx.DiGraph()
    G.add_nodes_from(data.columns)
    
    # Add prior edges if provided
    if prior_edges:
        G.add_edges_from(prior_edges)
    
    # Define the score function (BIC score)
    def score_function(X, parents, node):
        if len(parents) > max_parents:
            return float('-inf')
        if len(parents) == 0:
            return mutual_info_score(X[node], X[node])
        return mutual_info_score(X[parents], X[node])
    
    # Define the neighborhood function
    def get_neighborhood(G):
        for i in range(G.number_of_nodes()):
            for j in range(G.number_of_nodes()):
                if i != j:
                    yield (i, j)
    
    # Implement the Markov Chain Monte Carlo (MCMC) sampling
    current_score = sum(score_function(data, list(G.predecessors(node)), node) for node in G.nodes())
    
    for _ in range(iterations):
        i, j = random.choice(list(get_neighborhood(G)))
        
        # Propose a new graph
        G_new = G.copy()
        if G_new.has_edge(i, j):
            G_new.remove_edge(i, j)
        else:
            G_new.add_edge(i, j)
            
        # Check for cycles
        if nx.is_directed_acyclic_graph(G_new):
            new_score = sum(score_function(data, list(G_new.predecessors(node)), node) for node in G_new.nodes())
            
            # Accept or reject the new graph
            if new_score > current_score or random.random() < np.exp(new_score - current_score):
                G = G_new
                current_score = new_score
    
    # Implement dynamic connectivity
    dynamic_edges = []
    for edge in G.edges():
        source, target = edge
        correlation = data[source].rolling(window=20).corr(data[target])
        if correlation.std() > 0.1:  # Threshold for dynamic connectivity
            dynamic_edges.append(edge)
    
    # Add attributes to edges for dynamic connectivity
    nx.set_edge_attributes(G, {edge: {'dynamic': True} for edge in dynamic_edges})
    
    return G



def k2_algorithm(data, max_parents, prior_edges=None):
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
    prior_edges: Dict[Tuple[str, str], float] = None,
    allowed_connections: List[Tuple[str, str]] = None,
) -> float:
    if not parents:
        return -np.inf
    X = data[parents]
    y = data[node]

    # Check if all connections are allowed
    if allowed_connections:
        if not all((parent, node) in allowed_connections for parent in parents):
            return -np.inf

    # Incorporate prior knowledge if available
    prior_score = (
        sum(prior_edges.get((parent, node), 0) for parent in parents)
        if prior_edges
        else 0
    )

    # Compute the likelihood score (you may want to use a more sophisticated method)
    residuals = y - X.mean()
    likelihood_score = -np.sum(residuals**2)

    # Penalize complexity
    complexity_penalty = -0.5 * len(parents) * np.log(len(data))

    return likelihood_score + complexity_penalty + prior_score
