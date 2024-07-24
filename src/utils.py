import os
import json
from typing import Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd
from bayesian_network import HierarchicalBayesianNetwork
from bayesian_node import BayesianNode, CategoricalNode


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

    results["network_structure"] = self.explain_structure_extended()

    # Example individual for personalized recommendations
    example_individual = {
        'FS_Total_GM_Vol': self.data['FS_Total_GM_Vol'].mean(),
        'FS_Tot_WM_Vol': self.data['FS_Tot_WM_Vol'].mean(),
        'NEOFAC_O': self.data['NEOFAC_O'].mean()
    }
    results["personalized_recommendations_example"] = self.get_personalized_recommendations(example_individual)
    
    results["confidence_intervals"] = self.get_confidence_intervals()
    
    results["brain_stem_relationships"] = self.analyze_brain_stem_relationship()
    results["actionable_insights"] = self.generate_actionable_insights()
    results["personality_cognition_relationships"] = self.analyze_personality_cognition_relationship()
    results["age_dependent_relationships"] = self.analyze_age_dependent_relationships()
    
    # Enforce expected connections and refit if necessary
    self.enforce_expected_connections()
    results["updated_network_structure"] = self.explain_structure_extended()

    results["practical_implications"] = self.get_practical_implications()
    results["age_stratified_analysis"] = self.perform_age_stratified_analysis()
    results["unexpected_findings_explanations"] = self.explain_unexpected_findings()
    

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
        filename = f"bayesian_network_summary_{timestamp}.json"
    
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
        "future_research_directions": self.suggest_future_research(),
        "key_personality_cognition_findings": self._summarize_personality_cognition(results.get("personality_cognition_relationships")),
        "significant_age_dependent_changes": self._summarize_age_dependent_changes(results.get("age_dependent_relationships")),
    }
    
    if isinstance(self, HierarchicalBayesianNetwork):
        summary["hierarchical_structure"] = self.explain_hierarchical_structure()
    
    try:
        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary successfully written to {file_path}")
    except Exception as e:
        print(f"Error writing summary to file: {str(e)}")

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