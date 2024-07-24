import json
import os
from typing import Dict, Any, Tuple, Callable, List
from datetime import datetime
from functools import lru_cache
from joblib import Parallel, delayed
import logging

import pandas as pd
import numpy as np
from scipy import stats
import networkx as nx
import statsmodels.api as sm

from .bayesian_node import BayesianNode, CategoricalNode
from .structure_learning import learn_structure
from .parameter_fitting import fit_parameters


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianNetwork:
    def __init__(
        self,
        method="hill_climb",
        max_parents=2,
        iterations=300,
        categorical_columns=None,
        categories=None,
    ):
        self.method = method
        self.max_parents = max_parents
        self.iterations = iterations
        self.nodes = {}
        self.categorical_columns = categorical_columns or []
        self.categories = categories or {}
        self.prior_edges = []  # Initialize prior_edges as an empty list
        self.data = None  # Add this line

        for col in self.categorical_columns:
            if col in self.categories:
                self.nodes[col] = CategoricalNode(col, self.categories[col])
            else:
                self.nodes[col] = CategoricalNode(col, [])  # Empty categories for now

    @lru_cache(maxsize=128)
    def sample_node(self, node_name: str, size: int = 1) -> np.ndarray:
        sorted_nodes = self.topological_sort()
        samples = {node: None for node in sorted_nodes}

        for node in sorted_nodes:
            if node == node_name:
                break
            parent_values = {
                parent: samples[parent] for parent in self.nodes[node].parents
            }
            samples[node] = self.nodes[node].sample(size, parent_values)

        return samples[node_name]

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.apply(
            lambda col: (
                pd.Categorical(col).codes
                if col.name in self.categorical_columns
                else col
            )
        )

    def _learn_structure(self, data: pd.DataFrame):
        self.nodes = learn_structure(
            data,
            method=self.method,
            max_parents=self.max_parents,
            iterations=self.iterations,
            prior_edges=self.prior_edges,
        )

        # Enforce max_parents limit
        for node_name, node in self.nodes.items():
            if len(node.parents) > self.max_parents:
                print(
                    f"Warning: Node {node_name} has {len(node.parents)} parents. Limiting to {self.max_parents}."
                )
                node.parents = node.parents[: self.max_parents]

    def _create_nodes(self, data: pd.DataFrame):
        for column in data.columns:
            if column in self.categorical_columns:
                categories = data[column].astype("category").cat.categories.tolist()
                self.nodes[column] = CategoricalNode(column, categories)
            else:
                self.nodes[column] = BayesianNode(column)

    def add_edge(self, parent, child):
        if parent not in self.nodes or child not in self.nodes:
            raise ValueError("Both nodes must exist in the network")
        self.nodes[child].parents.append(self.nodes[parent])
        self.nodes[parent].children.append(self.nodes[child])

    def get_edges(self):
        edges = []
        for node_name, node in self.nodes.items():
            for parent in node.parents:
                edges.append((parent.name, node_name))
        return edges

    @lru_cache(maxsize=128)
    def _cached_node_log_likelihood(self, node_name, value, parent_values):
        node = self.nodes[node_name]
        return node.log_probability(value, parent_values)

    def log_likelihood(self, data: pd.DataFrame) -> float:
        log_probs = np.zeros(len(data))
        for node_name, node in self.nodes.items():
            parent_names = [parent.name for parent in node.parents]
            if parent_names:
                parent_values = data[parent_names].values
                log_probs += np.array(
                    [
                        node.log_probability(val, tuple(pv))
                        for val, pv in zip(data[node_name], parent_values)
                    ]
                )
            else:
                log_probs += np.array(
                    [node.log_probability(val) for val in data[node_name]]
                )
        return np.sum(log_probs)

    def compute_sensitivity(
        self, target_node: str, num_samples: int = 1000
    ) -> Dict[str, float]:
        sensitivity = {}
        target = self.nodes[target_node]

        if isinstance(target.distribution, tuple):
            base_mean, base_std = target.distribution
            base_mean = (
                np.mean(base_mean) if isinstance(base_mean, np.ndarray) else base_mean
            )
            base_std = (
                np.mean(base_std) if isinstance(base_std, np.ndarray) else base_std
            )
        else:
            base_mean, base_std = np.mean(target.distribution), np.std(
                target.distribution
            )

        base_std = max(base_std, 1e-10)  # Avoid division by zero

        for node_name, node in self.nodes.items():
            if node_name != target_node:
                if isinstance(node.distribution, tuple):
                    mean, std = node.distribution
                    mean = np.mean(mean) if isinstance(mean, np.ndarray) else mean
                    std = np.mean(std) if isinstance(std, np.ndarray) else std
                else:
                    mean, std = np.mean(node.distribution), np.std(node.distribution)

                std = max(std, 1e-10)  # Avoid using zero standard deviation

                # Calculate relative change
                delta_input = std / mean if np.abs(mean) > 1e-10 else 1

                if node in target.parents:
                    if isinstance(target.distribution, tuple):
                        coef = target.distribution[1][target.parents.index(node)]
                        coef = np.mean(coef) if isinstance(coef, np.ndarray) else coef
                    else:
                        coef = 1
                    delta_output = coef * delta_input
                else:
                    delta_output = 0.1 * delta_input  # Assume small indirect effect

                # Use relative change in sensitivity calculation
                sensitivity[node_name] = (
                    (delta_output / base_mean) / (delta_input / mean)
                    if np.abs(mean) > 1e-10
                    else 0
                )

        return sensitivity

    def get_key_relationships(self) -> List[Dict[str, Any]]:
        relationships = []
        for node_name, node in self.nodes.items():
            for parent in node.parents:
                strength = abs(self.compute_edge_strength(parent.name, node_name))
                relationships.append(
                    {"parent": parent.name, "child": node_name, "strength": strength}
                )
        sorted_relationships = sorted(
            relationships, key=lambda x: x["strength"], reverse=True
        )
        top_10_percent = sorted_relationships[: max(1, len(sorted_relationships) // 10)]
        return [
            {
                "parent": r["parent"],
                "child": r["child"],
                "strength": round(r["strength"], 2),
            }
            for r in top_10_percent
        ]

    def compute_edge_strength(self, parent_name: str, child_name: str) -> float:
        parent_node = self.nodes[parent_name]
        child_node = self.nodes[child_name]

        if (
            isinstance(child_node.distribution, tuple)
            and len(child_node.distribution) == 2
        ):
            coefficients = child_node.distribution[1]
            parent_index = child_node.parents.index(parent_node)
            return abs(coefficients[parent_index])
        else:
            # If the distribution is not in the expected format, return a default value
            return 0.0

    def get_novel_insights(self) -> List[str]:
        insights = []
        sensitivity = self.compute_sensitivity("CogFluidComp_Unadj")
        top_factors = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)[:3]
        for factor, value in top_factors:
            insights.append(
                f"Unexpectedly high influence of {factor} on cognitive fluid composite (sensitivity: {value:.2f})"
            )
        return insights


    def explain_key_relationships(self):
        explanations = []
        relationships = self.get_key_relationships()
        for rel in relationships:
            if (
                rel["parent"] == "FS_Total_GM_Vol"
                and rel["child"] == "CogFluidComp_Unadj"
            ):
                explanations.append(
                    f"Total gray matter volume strongly influences fluid cognitive ability (strength: {rel['strength']}). "
                    "This suggests that cognitive training programs should focus on activities that promote gray matter preservation, "
                    "such as complex problem-solving tasks and learning new skills."
                )
            elif (
                rel["parent"] == "FS_L_Hippo_Vol"
                and rel["child"] == "CogFluidComp_Unadj"
            ):
                explanations.append(
                    f"Left hippocampus volume is closely related to fluid cognitive ability (strength: {rel['strength']}). "
                    "This highlights the importance of memory-enhancing exercises in cognitive training, particularly those "
                    "that engage spatial navigation and episodic memory formation."
                )
            elif rel["parent"] == "NEOFAC_O" and rel["child"] == "CogCrystalComp_Unadj":
                explanations.append(
                    f"Openness to experience (NEOFAC_O) influences crystallized cognitive ability (strength: {rel['strength']}). "
                    "This suggests that encouraging curiosity and diverse learning experiences could enhance long-term cognitive performance."
                )
            # Add more specific explanations for other important relationships
        return explanations

    def get_performance_metrics(self):
        # Assuming we've already performed cross-validation
        mse = np.mean((self.y_true - self.y_pred) ** 2)
        r2 = 1 - (
            np.sum((self.y_true - self.y_pred) ** 2)
            / np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        )

        return {
            "Mean Squared Error": mse,
            "R-squared": r2,
            "Accuracy (within 10% of true value)": np.mean(
                np.abs(self.y_true - self.y_pred) / self.y_true < 0.1
            ),
        }

    def get_accuracy(self):
        if self.data is None:
            return 0.0
        X = self.data.drop(["CogFluidComp_Unadj", "CogCrystalComp_Unadj"], axis=1)
        y = self.data["CogFluidComp_Unadj"]
        y_binary = (y > y.median()).astype(int)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(random_state=42, max_iter=1000)
        predictions = cross_val_predict(model, X_scaled, y_binary, cv=5)
        return accuracy_score(y_binary, predictions)

    def get_precision(self):
        if self.data is None:
            return 0.0
        X = self.data.drop(["CogFluidComp_Unadj", "CogCrystalComp_Unadj"], axis=1)
        y = self.data["CogFluidComp_Unadj"]
        y_binary = (y > y.median()).astype(int)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(random_state=42, max_iter=1000)
        predictions = cross_val_predict(model, X_scaled, y_binary, cv=5)
        return precision_score(y_binary, predictions)

    def get_recall(self):
        if self.data is None:
            return 0.0
        X = self.data.drop(["CogFluidComp_Unadj", "CogCrystalComp_Unadj"], axis=1)
        y = self.data["CogFluidComp_Unadj"]
        y_binary = (y > y.median()).astype(int)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(random_state=42, max_iter=1000)
        predictions = cross_val_predict(model, X_scaled, y_binary, cv=5)
        return recall_score(y_binary, predictions)

    def get_log_likelihood(self, test_data):
        return self.log_likelihood(test_data)

    def get_posterior_predictive(self, new_data):
        # This is a placeholder - the actual implementation would depend on your specific model
        predictions = {}
        for node in self.nodes:
            if not self.nodes[node].parents:
                predictions[node] = self.nodes[node].sample(len(new_data))
            else:
                parent_values = {
                    parent.name: predictions[parent.name]
                    for parent in self.nodes[node].parents
                }
                predictions[node] = self.nodes[node].sample(
                    len(new_data), parent_values
                )
        return predictions



    def topological_sort(self) -> List[str]:
        graph = nx.DiGraph()
        for node_name, node in self.nodes.items():
            graph.add_node(node_name)
            for parent in node.parents:
                graph.add_edge(parent.name, node_name)

        return list(nx.topological_sort(graph))



    def get_confidence_intervals(self):
        ci_results = {}
        for node_name, node in self.nodes.items():
            if node.parents:
                X = self.data[[p.name for p in node.parents]]
                y = self.data[node_name]

                # Add a constant term to the predictors
                X = sm.add_constant(X)

                # Perform multiple linear regression
                model = sm.OLS(y, X).fit()

                ci_results[node_name] = {
                    "coefficients": model.params[1:].to_dict(),  # Exclude the intercept
                    "intercept": model.params.iloc[0],  # Get the intercept
                    "ci_lower": model.conf_int()[0].to_dict(),
                    "ci_upper": model.conf_int()[1].to_dict(),
                    "p_values": model.pvalues.to_dict(),
                }
        return ci_results

    def fit_transform(
        self,
        data: pd.DataFrame,
        prior_edges: List[tuple] = None,
        progress_callback: Callable[[float], None] = None,
    ):
        self.fit(data, prior_edges=prior_edges, progress_callback=progress_callback)
        return self.transform(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed_data = data.copy()
        for col in self.categorical_columns:
            if col in transformed_data:
                transformed_data[col] = (
                    transformed_data[col].astype("category").cat.codes
                )
        return transformed_data

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


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

    def fit(self, data: pd.DataFrame, level_constraints: Dict[str, List[str]] = None):
        preprocessed_data = self.preprocess_data(data)
        for level in self.levels:
            level_data = preprocessed_data[self.level_nodes[level]]
            allowed_parents = self.level_nodes[level]
            if level_constraints and level in level_constraints:
                allowed_parents += level_constraints[level]
            self._learn_structure(level_data, allowed_parents=allowed_parents)
        fit_parameters(self.nodes, preprocessed_data)

    def _learn_structure(self, data: pd.DataFrame, allowed_parents: List[str]):
        self.nodes = learn_structure(
            data,
            method=self.method,
            max_parents=self.max_parents,
            prior_edges=self.prior_edges,
        )
        for node in self.nodes.values():
            node.parents = [
                parent for parent in node.parents if parent.name in allowed_parents
            ]

        # Enforce the structure according to the allowed_parents
        for node_name, node in self.nodes.items():
            node.parents = [
                self.nodes[parent_name]
                for parent_name in allowed_parents
                if parent_name in self.nodes and parent_name in node.parents
            ]
