import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score, BDeuScore
import logging

from bayesian_node import BayesianNode, CategoricalNode

logger = logging.getLogger(__name__)

def learn_structure(data: pd.DataFrame, method: str = 'k2', max_parents: int = 2, iterations: int = 300, prior_edges: List[tuple] = None) -> List[Tuple[str, str]]:
    try:
        if method == 'hill_climb':
            raise ValueError("Hill climb method is not yet implemented in this function.")
        
        elif method == 'k2':
            # Convert prior_edges to a dictionary with a default score of 1.0
            prior_edges_dict = {edge: 1.0 for edge in (prior_edges or [])}

            # Initialize the K2 algorithm
            estimated_model = k2_algorithm(
                data,
                max_parents=max_parents,
                prior_edges=prior_edges_dict
            )

            # Convert the estimated model to a list of edges
            edges = []
            for node_name, node in estimated_model.items():
                for parent in node.parents:
                    edges.append((parent.name, node_name))

            logger.info(f"Structure learning complete. Learned {len(edges)} edges.")
            
            return edges

        else:
            raise ValueError(f"Unsupported method: {method}. Supported methods are 'hill_climb' and 'k2'.")

    except Exception as e:
        logger.error(f"Error in structure learning: {str(e)}")
        raise

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
