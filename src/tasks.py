from celery import Celery
import pandas as pd
import numpy as np
from bayesian_network import BayesianNetwork, HierarchicalBayesianNetwork


app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task
def fit_bayesian_network(data_dict, prior_edges, categorical_columns):
    data = pd.DataFrame(data_dict)
    model = BayesianNetwork(method='hill_climb', max_parents=4, categorical_columns=categorical_columns)
    model.fit(data, prior_edges=prior_edges)
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
    h_model = HierarchicalBayesianNetwork(levels=levels, method='hill_climb', max_parents=3, categorical_columns=categorical_columns)
    h_model.fit(data, level_constraints=level_constraints)
    return h_model.to_dict()