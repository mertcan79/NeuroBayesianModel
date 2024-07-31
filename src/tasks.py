from celery import Celery
from hierarchical_network import HierarchicalBayesianNetwork
import pandas as pd

app = Celery('NeuroBayesianModel',
             broker='redis://localhost:6379/0',
             backend='redis://localhost:6379/1')

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

@app.task
def fit_model(data_dict, categorical_columns, target_variable, prior_edges):
    data = pd.DataFrame(data_dict)
    model = HierarchicalBayesianNetwork(
        num_features=len(data.columns) - 1,
        max_parents=5,
        iterations=1500,
        categorical_columns=categorical_columns,
        target_variable=target_variable,
        prior_edges=prior_edges
    )
    model.fit(data)
    return model

@app.task
def compute_edge_weights(model):
    return model.compute_all_edge_probabilities()

@app.task
def explain_structure(model):
    return model.explain_structure_extended()

@app.task
def get_key_relationships(model):
    return model.get_key_relationships()

@app.task
def compute_sensitivities(model, target_variable):
    return model.compute_sensitivity(target_variable)

@app.task
def cross_validate(model, data_dict):
    data = pd.DataFrame(data_dict)
    return model.cross_validate_bayesian(data)

# Add more tasks for other analyses as needed