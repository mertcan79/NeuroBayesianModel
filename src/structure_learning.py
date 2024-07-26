import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
import statsmodels.api as sm

from bayesian_node import BayesianNode, CategoricalNode

logger = logging.getLogger(__name__)

def learn_structure(data, method="nsl", max_parents=4, iterations=300, prior_edges=None, alpha=0.05):
    if method == "nsl":
        return neurological_structure_learning(data, prior_edges, max_parents, alpha)
    else:
        raise ValueError(f"Unsupported structure learning method: {method}")

def neurological_structure_learning(data, prior_edges, max_parents=4, alpha=0.05):
    nodes = list(data.columns)
    edges = set(prior_edges) if prior_edges else set()

    def compute_bic(node, parents):
        X = data[parents]
        y = data[node]
        model = sm.OLS(y, sm.add_constant(X)).fit()
        bic = model.bic
        return bic
    
    for node in nodes:
        potential_parents = [n for n in nodes if n != node]
        current_parents = [p for p, c in edges if c == node]
        
        while len(current_parents) < max_parents:
            best_parent = None
            best_bic = float('inf')
            
            for parent in potential_parents:
                if parent not in current_parents:
                    new_parents = current_parents + [parent]
                    try:
                        bic = compute_bic(node, new_parents)
                        
                        if bic < best_bic:
                            best_parent = parent
                            best_bic = bic
                    except Exception as e:
                        print(f"Error computing BIC for {node} with parents {new_parents}: {e}")
            
            if best_parent is None or len(current_parents) >= max_parents:
                break
            
            current_parents.append(best_parent)
            edges.add((best_parent, node))
    return list(edges)



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
