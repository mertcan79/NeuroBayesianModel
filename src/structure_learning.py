import numpy as np
import pandas as pd
from typing import Dict, List
from bayesian_node import BayesianNode
from pgmpy.estimators import HillClimbSearch, BicScore

def learn_structure(data: pd.DataFrame, method: str = 'k2', max_parents: int = 5, prior_edges: Dict[tuple, float] = None) -> Dict[str, BayesianNode]:
    if method == 'k2':
        return k2_algorithm(data, max_parents, prior_edges)
    elif method == 'hill_climb':
        return hill_climb_algorithm(data, max_parents, prior_edges)
    else:
        raise ValueError(f"Unsupported structure learning method: {method}")

def k2_algorithm(data: pd.DataFrame, max_parents: int, prior_edges: Dict[tuple, float] = None) -> Dict[str, BayesianNode]:
    nodes = {node: BayesianNode(node) for node in data.columns}
    node_order = list(data.columns)

    for i, node in enumerate(node_order):
        parents = []
        old_score = score_node(data, node, parents, prior_edges)
        
        while len(parents) < max_parents:
            best_new_parent = None
            best_new_score = old_score

            for potential_parent in node_order[:i]:
                if potential_parent not in parents:  # Check if the parent is already added
                    new_parents = parents + [potential_parent]
                    new_score = score_node(data, node, new_parents, prior_edges)

                    if new_score > best_new_score:
                        best_new_parent = potential_parent
                        best_new_score = new_score

            if best_new_parent is None:
                break
            
            parents.append(best_new_parent)
            old_score = best_new_score

        nodes[node].parents = [nodes[p] for p in parents]
        for parent in parents:
            nodes[parent].children.append(nodes[node])

    return nodes

def hill_climb_algorithm(data: pd.DataFrame, max_parents: int, prior_edges: Dict[tuple, float] = None) -> Dict[str, BayesianNode]:
    hc = HillClimbSearch(data)
    model = hc.estimate(scoring_method=BicScore(data), max_indegree=max_parents)
    
    nodes = {node: BayesianNode(node) for node in data.columns}
    
    for edge in model.edges():
        parent, child = edge
        nodes[child].parents.append(nodes[parent])
        nodes[parent].children.append(nodes[child])
    
    # Incorporate prior edges if provided
    if prior_edges:
        for (parent, child), probability in prior_edges.items():
            if probability > 0.5 and nodes[parent] not in nodes[child].parents:
                nodes[child].parents.append(nodes[parent])
                nodes[parent].children.append(nodes[child])
    
    return nodes


def score_node(data: pd.DataFrame, node: str, parents: List[str], prior_edges: Dict[tuple, float] = None) -> float:
    # Implement your scoring function here (e.g., BIC score)
    # This is a placeholder implementation
    if not parents:
        return -np.inf
    X = data[parents]
    y = data[node]
    
    # Incorporate prior knowledge if available
    prior_score = sum(prior_edges.get((parent, node), 0) for parent in parents)
    
    # Compute the likelihood score (you may want to use a more sophisticated method)
    residuals = y - X.mean()
    likelihood_score = -np.sum(residuals**2)
    
    # Penalize complexity
    complexity_penalty = -0.5 * len(parents) * np.log(len(data))
    
    return likelihood_score + complexity_penalty + prior_score