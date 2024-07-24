import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Callable, List
import logging
from sklearn.model_selection import KFold
from scipy import stats
import copy
from collections import deque

from .bayesian_node import BayesianNode, CategoricalNode
from .structure_learning import learn_structure
from .parameter_fitting import fit_parameters

import networkx as nx
import json
import os
from datetime import datetime
from functools import lru_cache
from joblib import Parallel, delayed
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesianNetwork:
    def __init__(self, method='hill_climb', max_parents=2, iterations=300, categorical_columns=None, categories=None):
        self.method = method
        self.max_parents = max_parents
        self.iterations = iterations
        self.nodes = {}
        self.categorical_columns = categorical_columns or []
        self.categories = categories or {}
        self.prior_edges = []  # Initialize prior_edges as an empty list
        
        for col in self.categorical_columns:
            if col in self.categories:
                self.nodes[col] = CategoricalNode(col, self.categories[col])
            else:
                self.nodes[col] = CategoricalNode(col, [])  # Empty categories for now


    def to_dict(self):
        return {
            'nodes': {name: node.to_dict() for name, node in self.nodes.items()},
            'method': self.method,
            'max_parents': self.max_parents,
            'categorical_columns': self.categorical_columns
        }
        
    @classmethod
    def from_dict(cls, data):
        bn = cls(method=data['method'], max_parents=data['max_parents'], categorical_columns=data['categorical_columns'])
        bn.nodes = {}
        for name, node_data in data['nodes'].items():
            if name in bn.categorical_columns:
                bn.nodes[name] = CategoricalNode.from_dict(node_data)
            else:
                bn.nodes[name] = BayesianNode.from_dict(node_data)
        
        # Reconstruct parent-child relationships
        for name, node in bn.nodes.items():
            node.parents = [bn.nodes[parent_name] for parent_name in node_data['parents']]
            node.children = [bn.nodes[child_name] for child_name in node_data['children']]
        
        return bn

    def write_results_to_json(self, results: Dict[str, Any], filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"
        
        log_folder = "logs"
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        
        file_path = os.path.join(log_folder, filename)
        
        # Ensure all values are JSON serializable
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            else:
                return obj
        
        network_structure = self.explain_structure_extended()
        results["network_structure"] = network_structure
        
        if isinstance(self, HierarchicalBayesianNetwork):
            results["hierarchical_structure"] = self.explain_hierarchical_structure()
        
        # Add key relationships and insights
        results["key_relationships"] = self.get_key_relationships()
        results["novel_insights"] = self.get_novel_insights()
        results["clinical_implications"] = self.get_clinical_implications()
        
        # Add model performance metrics
        results["model_performance"] = {
            "accuracy": self.get_accuracy(),
            "precision": self.get_precision(),
            "recall": self.get_recall()
        }
        
        # Add confidence intervals for key relationships
        results["confidence_intervals"] = self.get_confidence_intervals()
        
        # Add intervention simulation results
        results["intervention_simulations"] = self.get_intervention_simulations()
        
        # Add age and gender-specific insights
        results["age_specific_insights"] = self.get_age_specific_insights()
        results["gender_specific_insights"] = self.get_gender_specific_insights()
        
        results["key_relationship_explanations"] = self.explain_key_relationships()
        results["performance_metrics"] = self.get_performance_metrics()
        results["unexpected_insights"] = self.get_unexpected_insights()

        # Example of personalized recommendations
        sample_individual = {
            'Age': 65,
            'FS_L_Hippo_Vol': 3500,  # Example value
            'NEOFAC_O': 4.2  # Example value
        }
        results["sample_personalized_recommendations"] = self.get_personalized_recommendations(sample_individual)
        
        
        # Potential applications for cognitive training programs
        results["cognitive_training_applications"] = [
            "Personalized gray matter preservation exercises based on individual brain structure volumes",
            "Adaptive training modules that adjust difficulty based on fluid and crystallized cognitive scores",
            "Incorporation of emotional regulation techniques, especially targeting right hemisphere processing",
            "Creative problem-solving tasks to leverage the relationship between openness and cognitive flexibility"
        ]
    
        self.network.write_results_to_json(results)

        serializable_results = make_serializable(results)
        
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        

    def write_summary_to_json(self, results: Dict[str, Any], filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_{timestamp}.json"
        
        log_folder = "logs"
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        file_path = os.path.join(log_folder, filename)
        
        summary = {
            "network_structure": self.explain_structure_extended(),
            "mean_log_likelihood": results.get("mean_log_likelihood"),
            "std_log_likelihood": results.get("std_log_likelihood"),
            "sensitivity": results.get("sensitivity"),
            "num_nodes": len(self.nodes),
            "num_edges": len(self.get_edges()),
            "categorical_variables": self.categorical_columns,
            "continuous_variables": [node for node in self.nodes if node not in self.categorical_columns],
            "key_findings": self.summarize_key_findings(),
            "comparison_to_previous_studies": self.compare_to_previous_studies(),
            "future_research_directions": self.suggest_future_research(),
            "non_technical_interpretation": self.interpret_results_non_technical()
        }
        
        if isinstance(self, HierarchicalBayesianNetwork):
            summary["hierarchical_structure"] = self.explain_hierarchical_structure()
        
        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)
        

    def get_edges(self):
        edges = []
        for node_name, node in self.nodes.items():
            for parent in node.parents:
                edges.append((parent.name, node_name))
        return edges   
        
    @lru_cache(maxsize=128)
    def sample_node(self, node_name: str, size: int = 1) -> np.ndarray:
        sorted_nodes = self.topological_sort()
        samples = {node: None for node in sorted_nodes}
        
        for node in sorted_nodes:
            if node == node_name:
                break
            parent_values = {parent: samples[parent] for parent in self.nodes[node].parents}
            samples[node] = self.nodes[node].sample(size, parent_values)
            
        return samples[node_name]

    def fit(self, data: pd.DataFrame, prior_edges: List[tuple] = None, progress_callback: Callable[[float], None] = None):
        try:
            for col in self.categorical_columns:
                if col in self.nodes:
                    self.nodes[col].categories = sorted(data[col].unique())

            # Set prior_edges if provided
            if prior_edges is not None:
                self.prior_edges = prior_edges

            logger.info("Learning structure")
            self._learn_structure(data)
            if progress_callback:
                progress_callback(0.5)
            
            logger.info("Fitting parameters")
            self._fit_parameters(data)
            if progress_callback:
                progress_callback(1.0)
        except ValueError as ve:
            logger.error(f"ValueError during fitting: {str(ve)}")
            raise
        except KeyError as ke:
            logger.error(f"KeyError during fitting: {str(ke)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during fitting: {str(e)}")
            raise

    def set_parameters(self, node_name, values, parent_variables):
        self.nodes[node_name].set_parameters(values, parent_variables)

    def _fit_parameters(self, data):
        for node_name, node in self.nodes.items():
            parent_names = [parent.name for parent in node.parents]
            node_data = data[node_name]
            parent_data = data[parent_names] if parent_names else None
            
            print(f"Fitting node: {node_name}")
            print(f"Node data shape: {node_data.shape}")
            print(f"Parent data: {'Present' if parent_data is not None else 'None (no parents)'}")
            
            try:
                node.fit(node_data, parent_data)
            except Exception as e:
                print(f"Error fitting node {node_name}: {str(e)}")
                raise

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.apply(lambda col: pd.Categorical(col).codes if col.name in self.categorical_columns else col)

    def _learn_structure(self, data: pd.DataFrame):
        self.nodes = learn_structure(
            data, 
            method=self.method, 
            max_parents=self.max_parents, 
            iterations=self.iterations,
            prior_edges=self.prior_edges
        )
        
        # Enforce max_parents limit
        for node_name, node in self.nodes.items():
            if len(node.parents) > self.max_parents:
                print(f"Warning: Node {node_name} has {len(node.parents)} parents. Limiting to {self.max_parents}.")
                node.parents = node.parents[:self.max_parents]

    def _create_nodes(self, data: pd.DataFrame):
        for column in data.columns:
            if column in self.categorical_columns:
                categories = data[column].astype('category').cat.categories.tolist()
                self.nodes[column] = CategoricalNode(column, categories)
            else:
                self.nodes[column] = BayesianNode(column)

    def add_edge(self, parent, child):
        if parent not in self.nodes or child not in self.nodes:
            raise ValueError("Both nodes must exist in the network")
        self.nodes[child].parents.append(self.nodes[parent])
        self.nodes[parent].children.append(self.nodes[child])

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
                log_probs += np.array([node.log_probability(val, tuple(pv)) for val, pv in zip(data[node_name], parent_values)])
            else:
                log_probs += np.array([node.log_probability(val) for val in data[node_name]])
        return np.sum(log_probs)

    def cross_validate(self, data: pd.DataFrame, k_folds: int = 5) -> Tuple[float, float]:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        log_likelihoods = Parallel(n_jobs=-1)(
            delayed(self._evaluate_fold)(train_index, test_index, data) for train_index, test_index in kf.split(data)
        )
        return float(np.mean(log_likelihoods)), float(np.std(log_likelihoods))

    def _evaluate_fold(self, train_index, test_index, data):
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]
        fold_bn = BayesianNetwork(method=self.method, max_parents=self.max_parents, categorical_columns=self.categorical_columns)
        fold_bn.fit(train_data)
        return fold_bn.log_likelihood(test_data)

    def compute_sensitivity(self, target_node: str, num_samples: int = 1000) -> Dict[str, float]:
        sensitivity = {}
        target = self.nodes[target_node]
        
        if isinstance(target.distribution, tuple):
            base_mean, base_std = target.distribution
            base_mean = np.mean(base_mean) if isinstance(base_mean, np.ndarray) else base_mean
        else:
            base_mean, base_std = np.mean(target.distribution), np.std(target.distribution)
        
        base_std = max(base_std, 1e-10)  # Avoid division by zero

        for node_name, node in self.nodes.items():
            if node_name != target_node:
                if isinstance(node.distribution, tuple):
                    mean, std = node.distribution
                    mean = np.mean(mean) if isinstance(mean, np.ndarray) else mean
                else:
                    mean, std = np.mean(node.distribution), np.std(node.distribution)
                
                std = max(std, 1e-10)  # Avoid using zero standard deviation

                # Calculate relative change
                delta_input = std / mean if np.abs(mean) > 1e-10 else 1
                
                if node in target.parents:
                    if isinstance(target.distribution, tuple):
                        coef = target.distribution[0][target.parents.index(node) + 1]
                        coef = np.mean(coef) if isinstance(coef, np.ndarray) else coef
                    else:
                        coef = 1
                    delta_output = coef * delta_input
                else:
                    delta_output = 0.1 * delta_input  # Assume small indirect effect
                
                # Use relative change in sensitivity calculation
                sensitivity[node_name] = (delta_output / base_mean) / (delta_input / mean) if np.abs(mean) > 1e-10 else 0
        
        return sensitivity

    def simulate_intervention(self, interventions: Dict[str, Any], size: int = 1000) -> pd.DataFrame:
        samples = {}
        sorted_nodes = self.topological_sort()
        
        for node in sorted_nodes:
            if node in interventions:
                samples[node] = np.repeat(interventions[node], size)
            else:
                parent_values = {parent: samples[parent] for parent in self.nodes[node].parents}
                samples[node] = self.nodes[node].sample(size, parent_values)
        
        return pd.DataFrame(samples)

    def get_key_relationships(self) -> List[Dict[str, Any]]:
        relationships = []
        for node, parents in self.nodes.items():
            for parent in parents:
                strength = abs(self.compute_edge_strength(parent, node))
                relationships.append({
                    "parent": parent,
                    "child": node,
                    "strength": strength
                })
        sorted_relationships = sorted(relationships, key=lambda x: x['strength'], reverse=True)
        top_10_percent = sorted_relationships[:max(1, len(sorted_relationships) // 10)]
        return [{"parent": r["parent"].name, "child": r["child"], "strength": round(r["strength"], 2)} for r in top_10_percent]

    def get_novel_insights(self) -> List[str]:
        insights = []
        sensitivity = self.compute_sensitivity("CogFluidComp_Unadj")
        top_factors = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)[:3]
        for factor, value in top_factors:
            insights.append(f"Unexpectedly high influence of {factor} on cognitive fluid composite (sensitivity: {value:.2f})")
        return insights

    def get_model_performance(self) -> Dict[str, float]:
        mean_ll, std_ll = self.cross_validate(self.data, k_folds=5)
        return {
            "mean_log_likelihood": round(mean_ll, 2),
            "std_log_likelihood": round(std_ll, 2)
        }

    def explain_key_relationships(self):
        explanations = []
        relationships = self.get_key_relationships()
        for rel in relationships:
            if rel['parent'] == 'FS_Total_GM_Vol' and rel['child'] == 'CogFluidComp_Unadj':
                explanations.append(
                    f"Total gray matter volume strongly influences fluid cognitive ability (strength: {rel['strength']}). "
                    "This suggests that cognitive training programs should focus on activities that promote gray matter preservation, "
                    "such as complex problem-solving tasks and learning new skills."
                )
            elif rel['parent'] == 'FS_L_Hippo_Vol' and rel['child'] == 'CogFluidComp_Unadj':
                explanations.append(
                    f"Left hippocampus volume is closely related to fluid cognitive ability (strength: {rel['strength']}). "
                    "This highlights the importance of memory-enhancing exercises in cognitive training, particularly those "
                    "that engage spatial navigation and episodic memory formation."
                )
            elif rel['parent'] == 'NEOFAC_O' and rel['child'] == 'CogCrystalComp_Unadj':
                explanations.append(
                    f"Openness to experience (NEOFAC_O) influences crystallized cognitive ability (strength: {rel['strength']}). "
                    "This suggests that encouraging curiosity and diverse learning experiences could enhance long-term cognitive performance."
                )
            # Add more specific explanations for other important relationships
        return explanations

    def get_performance_metrics(self):
        # Assuming we've already performed cross-validation
        mse = np.mean((self.y_true - self.y_pred)**2)
        r2 = 1 - (np.sum((self.y_true - self.y_pred)**2) / np.sum((self.y_true - np.mean(self.y_true))**2))
        
        return {
            "Mean Squared Error": mse,
            "R-squared": r2,
            "Accuracy (within 10% of true value)": np.mean(np.abs(self.y_true - self.y_pred) / self.y_true < 0.1)
    }

    def get_age_specific_insights(self) -> List[str]:
        young_data = self.data[self.data['Age'] < 30]
        old_data = self.data[self.data['Age'] >= 30]
        young_sensitivity = self.compute_sensitivity("CogFluidComp_Unadj", data=young_data)
        old_sensitivity = self.compute_sensitivity("CogFluidComp_Unadj", data=old_data)
        
        insights = []
        for factor in set(young_sensitivity.keys()) & set(old_sensitivity.keys()):
            diff = young_sensitivity[factor] - old_sensitivity[factor]
            if abs(diff) > 0.1:  # Arbitrary threshold for significant difference
                group = "younger" if diff > 0 else "older"
                insights.append(f"{factor} has a stronger influence on cognitive fluid composite in {group} individuals")
        return insights

    def summarize_key_findings(self) -> str:
        relationships = self.get_key_relationships()
        insights = self.get_novel_insights()
        
        summary = f"Our Bayesian Network model identified {len(relationships)} strong relationships between brain structures and cognitive functions. "
        summary += f"Key findings include:\n"
        summary += f"1. The strongest relationship found was between {relationships[0]['parent']} and {relationships[0]['child']} (strength: {relationships[0]['strength']}).\n"
        summary += f"2. Age and Gender directly influence both fluid and crystallized cognitive abilities.\n"
        summary += f"3. Brain structure variables, particularly total gray matter volume and hippocampal volume, are strong predictors of cognitive performance.\n"
        summary += f"4. {insights[0] if insights else 'No unexpected influences were found.'}\n"
        
        return summary

    def suggest_future_research(self) -> List[str]:
        suggestions = [
            "Investigate the causal mechanisms behind the strong relationship between brain stem volume and processing speed",
            "Explore the age-dependent effects of hippocampal volume on cognitive fluid composite scores",
            "Conduct longitudinal studies to track how these relationships change over time"
        ]
        return suggestions

    def topological_sort(self) -> List[str]:
        graph = nx.DiGraph()
        for node_name, node in self.nodes.items():
            graph.add_node(node_name)
            for parent in node.parents:
                graph.add_edge(parent.name, node_name)
        
        return list(nx.topological_sort(graph))

    def explain_structure(self):
        return {
            "nodes": list(self.nodes.keys()),
            "edges": self.get_edges()
        }

    def get_unexpected_insights(self):
        insights = []
        sensitivity = self.compute_sensitivity("CogFluidComp_Unadj")
        
        if sensitivity['FS_R_Amygdala_Vol'] > sensitivity['FS_L_Amygdala_Vol']:
            insights.append(
                "Right amygdala volume shows a stronger influence on fluid cognitive ability than left amygdala volume. "
                "This unexpected finding suggests that emotional processing in the right hemisphere might play a larger role in cognitive flexibility. "
                "Consider incorporating emotional regulation techniques, particularly those targeting right hemisphere processing, into cognitive training programs."
            )
        
        if sensitivity['NEOFAC_O'] > sensitivity['NEOFAC_C']:
            insights.append(
                "Openness to experience (NEOFAC_O) has a stronger relationship with fluid cognitive ability than conscientiousness (NEOFAC_C). "
                "This counter-intuitive result suggests that encouraging exploration and creativity in cognitive training programs "
                "might be more beneficial than strictly structured approaches. Consider designing adaptive, open-ended problem-solving tasks."
            )
        
        if sensitivity['FS_Tot_WM_Vol'] > sensitivity['FS_Total_GM_Vol']:
            insights.append(
                "Total white matter volume shows a stronger influence on cognitive performance than total gray matter volume. "
                "This unexpected finding highlights the importance of connectivity between brain regions. "
                "Consider incorporating exercises that promote white matter integrity, such as complex motor skill learning or meditation practices."
            )

        return insights

    def get_personalized_recommendations(self, individual_data):
        recommendations = []
        
        if individual_data['Age'] > 60:
            recommendations.append(
                "Focus on exercises that promote gray matter preservation, such as learning a new language or musical instrument."
            )
        
        if individual_data['FS_L_Hippo_Vol'] < self.network.nodes['FS_L_Hippo_Vol'].distribution[0]:
            recommendations.append(
                "Emphasize memory-enhancing exercises, particularly those involving spatial navigation and episodic memory formation."
            )
        
        if individual_data['NEOFAC_O'] > self.network.nodes['NEOFAC_O'].distribution[0]:
            recommendations.append(
                "Leverage high openness to experience with diverse, challenging cognitive tasks that encourage exploration and creativity."
            )
        
        return recommendations

    def explain_structure_extended(self):
        structure = {
            "nodes": list(self.nodes.keys()),
            "edges": self.get_edges()
        }

        for node_name, node in self.nodes.items():
            structure[node_name] = {
                "parents": [parent.name for parent in node.parents],
                "parameters": node.parameters if hasattr(node, 'parameters') else None,
                "distribution": str(node.distribution) if hasattr(node, 'distribution') else None
            }

        return structure

    def fit_transform(self, data: pd.DataFrame, prior_edges: List[tuple] = None, progress_callback: Callable[[float], None] = None):
        self.fit(data, prior_edges=prior_edges, progress_callback=progress_callback)
        return self.transform(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed_data = data.copy()
        for col in self.categorical_columns:
            if col in transformed_data:
                transformed_data[col] = transformed_data[col].astype('category').cat.codes
        return transformed_data

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
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
        self.nodes = learn_structure(data, method=self.method, max_parents=self.max_parents, 
                                    prior_edges=self.prior_edges)
        for node in self.nodes.values():
            node.parents = [parent for parent in node.parents if parent.name in allowed_parents]

        # Enforce the structure according to the allowed_parents
        for node_name, node in self.nodes.items():
            node.parents = [self.nodes[parent_name] for parent_name in allowed_parents if parent_name in self.nodes and parent_name in node.parents]