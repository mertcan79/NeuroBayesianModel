from celery import Celery
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from bayesian_network import BayesianNetwork, HierarchicalBayesianNetwork

app = Celery('NeuroBayesianModel',
             broker='redis://localhost:6379/0',
             backend='redis://localhost:6379/1')

app.conf.update(
    task_serializer='pickle',
    accept_content=['pickle', 'json'],
    result_serializer='pickle',
    timezone='UTC',
    enable_utc=True,
)

@app.task
def fit_bayesian_network(data_dict: Dict[str, List], prior_edges: List[tuple], categorical_columns: List[str], method: str = 'k2', max_parents: int = 2, iterations: int = 100):
    data = pd.DataFrame(data_dict)
    model = BayesianNetwork(method=method, max_parents=max_parents, iterations=iterations, categorical_columns=categorical_columns)
    model.fit(data, prior_edges=prior_edges)
    return model.to_dict()

@app.task
def compute_network_metrics(model_dict: Dict[str, Any], data_dict: Dict[str, List]):
    model = BayesianNetwork.from_dict(model_dict)
    data = pd.DataFrame(data_dict)
    
    results = {
        'edge_probabilities': model.compute_edge_probabilities(),
        'key_relationships': model.get_key_relationships(),
        'novel_insights': model.get_novel_insights(),
    }
    
    return results

@app.task
def perform_sensitivity_analysis(model_dict: Dict[str, Any], target_nodes: List[str], num_samples: int = 1000):
    model = BayesianNetwork.from_dict(model_dict)
    sensitivities = {}
    for target_node in target_nodes:
        sensitivities[target_node] = model.compute_sensitivity(target_node, num_samples)
    return sensitivities

@app.task
def simulate_interventions(model_dict: Dict[str, Any], data_dict: Dict[str, List], interventions: Dict[str, Any], size: int = 1000):
    model = BayesianNetwork.from_dict(model_dict)
    data = pd.DataFrame(data_dict)
    simulated_data = model.simulate_intervention(data, interventions, size)
    return simulated_data.to_dict(orient='list')

@app.task
def fit_hierarchical_bayesian_network(data_dict: Dict[str, List], levels: List[str], level_constraints: Dict[str, List[str]], categorical_columns: List[str], method: str = 'k2', max_parents: int = 2, iterations: int = 100):
    data = pd.DataFrame(data_dict)
    h_model = HierarchicalBayesianNetwork(levels=levels, method=method, max_parents=max_parents, iterations=iterations, categorical_columns=categorical_columns)
    h_model.fit(data, level_constraints=level_constraints)
    return h_model.to_dict()

@app.task
def analyze_age_gender_effects(model_dict: Dict[str, Any], data_dict: Dict[str, List]):
    model = BayesianNetwork.from_dict(model_dict)
    data = pd.DataFrame(data_dict)
    
    age_insights = model.get_age_specific_insights(data)
    gender_insights = model.get_gender_specific_insights(data)
    
    return {
        'age_specific_insights': age_insights,
        'gender_specific_insights': gender_insights
    }