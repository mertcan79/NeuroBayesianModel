from celery import Celery
from hierarchical_network import HierarchicalBayesianNetwork
import pandas as pd
import numpy as np
import json
import jax.numpy as jnp
from jax import Array
import jax


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

def convert_to_serializable(obj):
    if isinstance(obj, (np.ndarray, jax.numpy.ndarray)):
        return np.array(obj).tolist()
    elif isinstance(obj, jax.Array):
        return np.array(obj).tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif hasattr(obj, 'item'):  # This will handle numpy scalar types like float32
        return obj.item()
    return obj

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, jax.numpy.ndarray, jax.Array)):
            return np.array(obj).tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_json_encoder=CustomJSONEncoder
)

def reconstruct_model(model_data):
    model = HierarchicalBayesianNetwork(
        num_features=model_data['num_features'],
        max_parents=3,
        iterations=750,
        categorical_columns=model_data['categorical_columns'],
        target_variable=model_data['target_variable'],
        prior_edges=model_data['prior_edges']
    )
    model.samples = {k: jax.numpy.array(v) for k, v in model_data['samples'].items()}
    model.edge_weights = jax.numpy.array(model_data['edge_weights'])
    model.data = pd.DataFrame(model_data['data'])
    return model

@app.task
def fit_model(data_dict, categorical_columns, target_variable, prior_edges):
    data = pd.DataFrame(data_dict)
    model = HierarchicalBayesianNetwork(
        num_features=len(data.columns) - 1,
        max_parents=3,
        iterations=750,
        categorical_columns=categorical_columns,
        target_variable=target_variable,
        prior_edges=prior_edges
    )
    model.fit(data)
    
    return convert_to_serializable({
        'samples': model.samples,
        'edge_weights': model.edge_weights,
        'data': model.data.to_dict(),
        'categorical_columns': model.categorical_columns,
        'target_variable': model.target_variable,
        'prior_edges': model.prior_edges,
        'num_features': model.num_features
    })

@app.task
def compute_edge_weights(model_data):
    model = reconstruct_model(model_data)
    result = model.compute_all_edge_probabilities()
    print(f"Type of result: {type(result)}")  # Debug print
    print(f"Contents of result: {result}")  # Debug print
    serializable_result = convert_to_serializable(result)
    return serializable_result

@app.task
def explain_structure(model_data):
    model = reconstruct_model(model_data)
    return convert_to_serializable(model.explain_structure_extended())

@app.task
def get_key_relationships(model_data):
    model = reconstruct_model(model_data)
    return convert_to_serializable(model.get_key_relationships())

@app.task
def compute_sensitivities(model_data, target_variable):
    model = reconstruct_model(model_data)
    return convert_to_serializable(model.compute_sensitivity(target_variable))

@app.task
def cross_validate(model_data, data_dict):
    model = reconstruct_model(model_data)
    data = pd.DataFrame(data_dict)
    result = model.cross_validate_bayesian(data)
    return convert_to_serializable(result)

@app.task
def get_clinical_implications(model_data):
    model = reconstruct_model(model_data)
    return convert_to_serializable(model.get_clinical_implications())

@app.task
def analyze_age_dependent_relationships(model_data, data_dict, age_column, target_variable):
    model = reconstruct_model(model_data)
    data = pd.DataFrame(data_dict)
    result = model.analyze_age_dependent_relationships(age_column, target_variable)
    return convert_to_serializable(result)

@app.task
def perform_interaction_effects_analysis(model_data, target_variable):
    model = reconstruct_model(model_data)
    result = model.perform_interaction_effects_analysis(target_variable)
    return convert_to_serializable(result)

@app.task
def perform_counterfactual_analysis(model_data, data_dict, interventions, target_variable):
    model = reconstruct_model(model_data)
    data = pd.DataFrame(data_dict)
    return convert_to_serializable(model.perform_counterfactual_analysis(interventions, target_variable))

@app.task
def perform_sensitivity_analysis(model_data, target_variable):
    model = reconstruct_model(model_data)
    return convert_to_serializable(model.perform_sensitivity_analysis(target_variable))

@app.task
def fit_nonlinear_model(model_data, data_dict, target_variable):
    model = reconstruct_model(model_data)
    data = pd.DataFrame(data_dict)
    model.fit_nonlinear(data, target_variable)
    
    return convert_to_serializable({
        'samples': model.samples,
        'edge_weights': model.edge_weights,
        'data': model.data.to_dict(),
        'categorical_columns': model.categorical_columns,
        'target_variable': model.target_variable,
        'prior_edges': model.prior_edges,
        'num_features': model.num_features
    })

@app.task
def bayesian_model_comparison(model_data, data_dict, models):
    model = reconstruct_model(model_data)
    data = pd.DataFrame(data_dict)
    return convert_to_serializable(model.bayesian_model_comparison(data, models))

@app.task
def analyze_age_related_changes(model_data, data_dict, age_column, cognitive_measures):
    model = reconstruct_model(model_data)
    data = pd.DataFrame(data_dict)
    return convert_to_serializable(model.analyze_age_related_changes(age_column, cognitive_measures))

@app.task
def compare_performance(model_data, data_dict, target_variable):
    model = reconstruct_model(model_data)
    data = pd.DataFrame(data_dict)
    return convert_to_serializable(model.compare_performance(model, data, target_variable))