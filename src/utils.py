import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from bayesian_network import HierarchicalBayesianNetwork
from bayesian_node import BayesianNode, CategoricalNode
from .bayesian_network import BayesianNetwork
from typing import Dict, Any, List, Tuple
from .insights import (
    explain_structure_extended, summarize_key_findings,
    summarize_personality_cognition, summarize_age_dependent_changes, get_key_relationships,
    perform_age_stratified_analysis, get_practical_implications, get_age_specific_insights,
    get_gender_specific_insights, explain_key_relationships, get_unexpected_insights, get_personalized_recommendations,
    analyze_brain_stem_relationship, generate_actionable_insights, analyze_personality_cognition_relationship, analyze_age_dependent_relationships,
    explain_unexpected_findings, get_novel_insights, get_clinical_implications, get_performance_metrics

)
#from .bayesian_network import get_confidence_intervals, get_performance_metrics, get_intervention_simulations

def write_results_to_json(results: Dict[str, Any], filename: str = None):
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

    
    #if isinstance(HierarchicalBayesianNetwork):
    #    results["hierarchical_structure"] = explain_hierarchical_structure()
    
    # Add key relationships and insights
    results["key_relationships"] = get_key_relationships()
    results["novel_insights"] = get_novel_insights()
    results["clinical_implications"] = get_clinical_implications()
    
    # Add model performance metrics
    results["model_performance"] = {
        "accuracy": get_accuracy(),
        "precision": get_precision(),
        "recall": get_recall()
    }
    
    # Add confidence intervals for key relationships
    results["confidence_intervals"] = get_confidence_intervals()
    
    # Add intervention simulation results
    results["intervention_simulations"] = get_intervention_simulations()
    
    # Add age and gender-specific insights
    results["age_specific_insights"] = get_age_specific_insights()
    results["gender_specific_insights"] = get_gender_specific_insights()
    
    results["key_relationship_explanations"] = explain_key_relationships()
    results["performance_metrics"] = get_performance_metrics()
    results["unexpected_insights"] = get_unexpected_insights()

    results["network_structure"] = explain_structure_extended()

    # Example individual for personalized recommendations
    example_individual = {
        'FS_Total_GM_Vol': data['FS_Total_GM_Vol'].mean(),
        'FS_Tot_WM_Vol': data['FS_Tot_WM_Vol'].mean(),
        'NEOFAC_O': data['NEOFAC_O'].mean()
    }
    results["personalized_recommendations_example"] = get_personalized_recommendations(example_individual)
    
    results["confidence_intervals"] = get_confidence_intervals()
    
    results["brain_stem_relationships"] = analyze_brain_stem_relationship()
    results["actionable_insights"] = generate_actionable_insights()
    results["personality_cognition_relationships"] = analyze_personality_cognition_relationship()
    results["age_dependent_relationships"] = analyze_age_dependent_relationships()
    
    # Enforce expected connections and refit if necessary
    enforce_expected_connections()
    results["updated_network_structure"] = explain_structure_extended()

    results["practical_implications"] = get_practical_implications()
    results["age_stratified_analysis"] = perform_age_stratified_analysis()
    results["unexpected_findings_explanations"] = explain_unexpected_findings()
    

    # Potential applications for cognitive training programs
    results["cognitive_training_applications"] = [
        "Personalized gray matter preservation exercises based on individual brain structure volumes",
        "Adaptive training modules that adjust difficulty based on fluid and crystallized cognitive scores",
        "Incorporation of emotional regulation techniques, especially targeting right hemisphere processing",
        "Creative problem-solving tasks to leverage the relationship between openness and cognitive flexibility"
    ]

    network.write_results_to_json(results)

    serializable_results = make_serializable(results)
    
    with open(file_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def write_summary_to_json(network, results: Dict[str, Any], filename: str = None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bayesian_network_summary_{timestamp}.json"
    
    log_folder = "logs"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    file_path = os.path.join(log_folder, filename)

    summary = {
        "network_structure": network.explain_structure_extended(),
        "mean_log_likelihood": results.get("mean_log_likelihood"),
        "std_log_likelihood": results.get("std_log_likelihood"),
        "sensitivity": results.get("sensitivity"),
        "num_nodes": len(network.nodes),
        "num_edges": len(network.get_edges()),
        "categorical_variables": network.categorical_columns,
        "continuous_variables": [node for node in network.nodes if node not in network.categorical_columns],
        "key_findings": network.summarize_key_findings(),
        "future_research_directions": network.suggest_future_research(),
        "key_personality_cognition_findings": summarize_personality_cognition(results.get("personality_cognition_relationships")),
        "significant_age_dependent_changes": summarize_age_dependent_changes(results.get("age_dependent_relationships")),
    }
    
    try:
        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary successfully written to {file_path}")
    except Exception as e:
        print(f"Error writing summary to file: {str(e)}")

def network_to_dict(nodes, method, max_parents, categorical_columns):
    return {
        'nodes': {name: node.to_dict() for name, node in nodes.items()},
        'method': method,
        'max_parents': max_parents,
        'categorical_columns': categorical_columns
    }

def network_from_dict(data):


    bn = BayesianNetwork(method=data['method'], max_parents=data['max_parents'], categorical_columns=data['categorical_columns'])
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