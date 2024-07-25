import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score, BDeuScore
import logging

from bayesian_node import BayesianNode, CategoricalNode

logger = logging.getLogger(__name__)

def learn_structure(data: pd.DataFrame, method: str = 'k2', max_parents: int = 2, iterations: int = 300, prior_edges: List[tuple] = None) -> Dict[str, BayesianNode]:
    """
    Learn the structure of a Bayesian Network from data.

    :param data: DataFrame containing the data
    :param method: Structure learning method ('hill_climb' or 'k2')
    :param max_parents: Maximum number of parents for any node
    :param iterations: Maximum number of iterations for the K2 algorithm
    :param prior_edges: List of tuples representing prior edges to include in the network
    :return: Dictionary of BayesianNode objects representing the learned network structure
    """
    try:
        if method == 'hill_climb':
            # Existing hill_climb code
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

            # Convert pgmpy model to our custom format
            nodes = {}
            for node in estimated_model:
                nodes[node] = BayesianNode(node)

            for node, node_obj in nodes.items():
                for parent in node_obj.parents:
                    nodes[parent].add_child(node_obj)
                    node_obj.add_parent(nodes[parent])

            logger.info(f"Structure learning complete. Learned {len(nodes)} nodes and {len(estimated_model)} edges.")
            
            return nodes

        else:
            raise ValueError(f"Unsupported method: {method}. Supported methods are 'hill_climb' and 'k2'.")

    except Exception as e:
        logger.error(f"Error in structure learning: {str(e)}")
        raise

def k2_algorithm(
    data: pd.DataFrame,
    max_parents: int,
    prior_edges: Dict[Tuple[str, str], float] = None,
    allowed_connections: List[Tuple[str, str]] = None,
) -> Dict[str, BayesianNode]:
    nodes = {node: BayesianNode(node) for node in data.columns}
    node_order = list(data.columns)

    for i, node in enumerate(node_order):
        parents = []
        old_score = score_node(data, node, parents, prior_edges, allowed_connections)

        while len(parents) < max_parents:
            best_new_parent = None
            best_new_score = old_score

            for potential_parent in node_order[:i]:
                if potential_parent not in parents:
                    if (
                        allowed_connections
                        and (potential_parent, node) not in allowed_connections
                    ):
                        continue
                    new_parents = parents + [potential_parent]
                    new_score = score_node(
                        data, node, new_parents, prior_edges, allowed_connections
                    )

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
