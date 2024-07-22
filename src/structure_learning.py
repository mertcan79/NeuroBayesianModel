import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pgmpy.estimators import HillClimbSearch, BicScore
from bayesian_node import BayesianNode, CategoricalNode

def learn_structure(data, method="hill_climb", max_parents=5, prior_edges=None, categorical_columns=None, existing_nodes=None):
    categorical_columns = categorical_columns or []
    existing_nodes = existing_nodes or {}
    prior_edges = prior_edges or []
    
    # Create a copy of the data with categorical columns encoded
    data_encoded = data.copy()

    if method == "hill_climb":
        hc = HillClimbSearch(data_encoded)
        
        # Create a custom scoring function that incorporates prior edges
        def custom_score(model):
            base_score = BicScore(data_encoded).score(model)
            edge_bonus = sum(1 for edge in prior_edges if model.has_edge(*edge))
            return base_score + edge_bonus * 10  # Adjust the bonus multiplier as needed

        # Estimate the structure using the custom scoring function
        model = hc.estimate(scoring_method=custom_score, max_indegree=max_parents)

        nodes = existing_nodes.copy()
        for column in data.columns:
            if column not in nodes:
                if column in categorical_columns:
                    categories = sorted(data[column].unique())
                    nodes[column] = CategoricalNode(column, categories)
                else:
                    nodes[column] = BayesianNode(column)

        # Add edges based on the learned structure
        for node in model.nodes():
            for parent in model.get_parents(node):
                if parent in nodes and node in nodes:
                    nodes[node].parents.append(nodes[parent])
                    nodes[parent].children.append(nodes[node])

        # Add prior edges that are not already in the structure
        for parent, child in prior_edges:
            if parent in nodes and child in nodes and nodes[parent] not in nodes[child].parents:
                nodes[child].parents.append(nodes[parent])
                nodes[parent].children.append(nodes[child])

        return nodes
    else:
        raise ValueError(f"Unsupported structure learning method: {method}")

def _learn_structure(self, data: pd.DataFrame, prior_edges: List[tuple] = None):
    self.nodes = learn_structure(data, method=self.method, max_parents=self.max_parents, 
                                 prior_edges=self.prior_edges, categorical_columns=self.categorical_columns,
                                 existing_nodes=self.nodes)
    if prior_edges:
        for edge in prior_edges:
            if edge[0] in self.nodes and edge[1] in self.nodes:
                if self.nodes[edge[0]] not in self.nodes[edge[1]].parents:
                    self.nodes[edge[1]].parents.append(self.nodes[edge[0]])
                    self.nodes[edge[0]].children.append(self.nodes[edge[1]])

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
