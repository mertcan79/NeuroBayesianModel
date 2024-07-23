import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score, BDeuScore
import logging

from .bayesian_node import BayesianNode, CategoricalNode

logger = logging.getLogger(__name__)


def learn_structure(data: pd.DataFrame, method: str = 'hill_climb', max_parents: int = 4, iterations: int = 1000, prior_edges: List[tuple] = None) -> Dict[str, BayesianNode]:
    """
    Learn the structure of a Bayesian Network from data.
    
    :param data: DataFrame containing the data
    :param method: Structure learning method (currently only 'hill_climb' is supported)
    :param max_parents: Maximum number of parents for any node
    :param iterations: Maximum number of iterations for the Hill Climbing algorithm
    :param prior_edges: List of tuples representing prior edges to include in the network
    :return: Dictionary of BayesianNode objects representing the learned network structure
    """
    try:
        if method != 'hill_climb':
            raise ValueError(f"Unsupported method: {method}. Only 'hill_climb' is currently supported.")

        # Initialize the Hill Climbing search
        hc = HillClimbSearch(data)
        bdeu_score = BDeuScore(data, equivalent_sample_size=10)

        # Create a blacklist of edges (if needed)
        blacklist = []

        # Create a whitelist of edges from prior_edges
        whitelist = prior_edges if prior_edges else []

        logger.info(f"Starting structure learning with max_parents={max_parents}, iterations={iterations}")
        
        # Estimate the model structure
        estimated_model = hc.estimate(
            scoring_method=bdeu_score,
            max_indegree=max_parents,
            black_list=blacklist,
            white_list=whitelist,
            epsilon=1e-4,
            max_iter=iterations
        )

        if prior_edges:
            for edge in prior_edges:
                if edge not in estimated_model.edges():
                    estimated_model.add_edge(*edge)

        # Convert pgmpy model to our custom format
        nodes = {}
        for node in estimated_model.nodes():
            nodes[node] = BayesianNode(node)

        for edge in estimated_model.edges():
            parent, child = edge
            nodes[child].add_parent(nodes[parent])
            nodes[parent].add_child(nodes[child])

        logger.info(f"Structure learning complete. Learned {len(nodes)} nodes and {len(estimated_model.edges())} edges.")
        
        return nodes

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

def hill_climb_algorithm(
    data: pd.DataFrame,
    max_parents: int,
    prior_edges: Dict[Tuple[str, str], float] = None,
    allowed_connections: List[Tuple[str, str]] = None,
) -> Dict[str, BayesianNode]:
    hc = HillClimbSearch(data)

    def modified_bic_score(model):
        score = BicScore(data).score(model)
        if allowed_connections:
            for edge in model.edges():
                if edge not in allowed_connections:
                    return -np.inf
        return score

    model = hc.estimate(scoring_method=modified_bic_score, max_indegree=max_parents)

    nodes = {node: BayesianNode(node) for node in data.columns}

    for edge in model.edges():
        parent, child = edge
        nodes[child].parents.append(nodes[parent])
        nodes[parent].children.append(nodes[child])

    # Incorporate prior edges if provided
    if prior_edges:
        for (parent, child), probability in prior_edges.items():
            if probability > 0.5 and nodes[parent] not in nodes[child].parents:
                if allowed_connections and (parent, child) in allowed_connections:
                    nodes[child].parents.append(nodes[parent])
                    nodes[parent].children.append(nodes[child])

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
