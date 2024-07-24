from celery import Celery
import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BayesianEstimator, BicScore
from pgmpy.models import BayesianNetwork as PgmpyBN
from pgmpy.estimators import K2Score

from .bayesian_network import BayesianNetwork, HierarchicalBayesianNetwork

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
def fit_bayesian_network(data_dict, prior_edges, categorical_columns):
    data = pd.DataFrame(data_dict)
    
    # Use pgmpy's HillClimbSearch with K2Score
    hc = HillClimbSearch(data)
    k2_score = K2Score(data)
    
    # Create a blacklist of edges that are not in prior_edges
    blacklist = []
    if prior_edges:
        for node1 in data.columns:
            for node2 in data.columns:
                if node1 != node2 and (node1, node2) not in prior_edges and (node2, node1) not in prior_edges:
                    blacklist.append((node1, node2))
                    blacklist.append((node2, node1))
    
    # Define the best model, incorporating prior edges through the blacklist
    best_model = hc.estimate(
        scoring_method=k2_score,
        max_indegree=4,
        black_list=blacklist if blacklist else None
    )
    
    # Convert pgmpy model to our BayesianNetwork format
    model = BayesianNetwork(method='k2', max_parents=2, iterations= 300, categorical_columns=categorical_columns)
    
    # Add edges from best_model and prior_edges
    for edge in best_model.edges():
        model.add_edge(*edge)
    for edge in prior_edges:
        if edge not in best_model.edges():
            model.add_edge(*edge)
    
    # Fit parameters using pgmpy
    pgmpy_model = PgmpyBN(model.get_edges())
    pgmpy_model.fit(data, estimator=BayesianEstimator, prior_type='BDeu')
    
    # Transfer parameters from pgmpy model to our model
    for node in pgmpy_model.nodes():
        cpd = pgmpy_model.get_cpds(node)
        if cpd.variable in model.nodes:
            model.nodes[cpd.variable].params = cpd.values.tolist()
            model.nodes[cpd.variable].parents = [model.nodes[var] for var in cpd.variables[1:]]
    
    return model.to_dict()

@app.task
def compute_sensitivity(model_dict, target_node, num_samples):
    model = BayesianNetwork.from_dict(model_dict)
    return model.compute_sensitivity(target_node, num_samples)

@app.task
def cross_validate(model_dict, data_dict, k_folds):
    model = BayesianNetwork.from_dict(model_dict)
    data = pd.DataFrame(data_dict)
    return model.cross_validate(data, k_folds)

@app.task
def fit_hierarchical_bayesian_network(data_dict, levels, level_constraints, categorical_columns):
    data = pd.DataFrame(data_dict)
    h_model = HierarchicalBayesianNetwork(levels=levels, method='hill_climb', max_parents=4, categorical_columns=categorical_columns)
    h_model.fit(data, level_constraints=level_constraints)
    return h_model.to_dict()
