from celery import Celery
from hierarchical_network import HierarchicalBayesianNetwork
import pandas as pd
import numpy as np
import json
import jax

from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

app = Celery('NeuroBayesianModel',
             broker='redis://localhost:6379/0',
             backend='redis://localhost:6379/1')

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_json_encoder=lambda obj: json.dumps(obj, cls=CustomJSONEncoder),
    worker_prefetch_multiplier=2,
    task_acks_late=True,
    task_time_limit=3600,
    task_soft_time_limit=3540,
    task_routes={
        'compute_intensive_task': {'queue': 'compute'},
        'io_intensive_task': {'queue': 'io'},
    },
    task_track_started=True,
    task_retry_limit=3,
    task_default_retry_delay=300,
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
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, jax.numpy.ndarray, jax.Array)):
            return np.array(obj).tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)

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

@app.task(bind=True, max_retries=3, default_retry_delay=300)
def fit_model(self, data_dict, categorical_columns, target_variable, prior_edges):
    try:
        data = pd.DataFrame(data_dict)
        logger.info(f"Creating HierarchicalBayesianNetwork with {len(data.columns)} features")
        model = HierarchicalBayesianNetwork(
            num_features=len(data.columns) - 1,
            max_parents=3,
            iterations=750,
            categorical_columns=categorical_columns,
            target_variable=target_variable,
            prior_edges=prior_edges
        )
        logger.info(f"Fitting model with {len(data)} samples")
        model.fit(data)

        logger.info("Model fitted successfully")
        logger.info(f"Samples keys: {model.samples.keys()}")

        result = {
            'samples': model.samples,
            'edge_weights': model.edge_weights,
            'data': model.data.to_dict(),
            'categorical_columns': model.categorical_columns,
            'target_variable': model.target_variable,
            'prior_edges': model.prior_edges,
            'num_features': model.num_features
        }

        return convert_to_serializable(result)
    except Exception as e:
        logger.error(f"Error in fit_model: {str(e)}")
        logger.exception("Exception details:")
        raise self.retry(exc=e)

@app.task(bind=True, max_retries=3, default_retry_delay=300)
def compute_edge_weights(self, model_data):
    try:
        model = reconstruct_model(model_data)
        result = model.compute_all_edge_probabilities()
        serializable_result = convert_to_serializable(result)
        return serializable_result
    except Exception as e:
        logger.error(f"Error in compute_edge_weights: {str(e)}")
        logger.exception("Exception details:")
        raise self.retry(exc=e)

@app.task(bind=True, max_retries=3, default_retry_delay=300)
def explain_structure(self, model_data):
    try:
        model = reconstruct_model(model_data)
        return convert_to_serializable(model.explain_structure_extended())
    except Exception as e:
        logger.error(f"Error in explain_structure: {str(e)}")
        logger.exception("Exception details:")
        raise self.retry(exc=e)

@app.task(bind=True, max_retries=3, default_retry_delay=300)
def get_key_relationships(self, model_data):
    try:
        model = reconstruct_model(model_data)
        return convert_to_serializable(model.get_key_relationships())
    except Exception as e:
        logger.error(f"Error in get_key_relationships: {str(e)}")
        logger.exception("Exception details:")
        raise self.retry(exc=e)

@app.task(bind=True, max_retries=3, default_retry_delay=300)
def compute_sensitivities(self, model_data, target_variable):
    try:
        model = reconstruct_model(model_data)
        return convert_to_serializable(model.compute_sensitivity(target_variable))
    except Exception as e:
        logger.error(f"Error in compute_sensitivities: {str(e)}")
        logger.exception("Exception details:")
        raise self.retry(exc=e)

@app.task(bind=True, max_retries=3, default_retry_delay=300)
def cross_validate(self, model_data, data_dict):
    try:
        model = reconstruct_model(model_data)
        data = pd.DataFrame(data_dict)
        logger.info(f"Starting cross-validation with {len(data)} samples")
        result = model.cross_validate_bayesian(data)
        logger.info("Cross-validation completed successfully")
        return convert_to_serializable(result)
    except Exception as e:
        logger.error(f"Error in cross_validate: {str(e)}")
        logger.exception("Exception details:")
        raise self.retry(exc=e)

@app.task(bind=True, max_retries=3, default_retry_delay=300)
def perform_interaction_effects_analysis(self, model_data, target_variable):
    try:
        model = reconstruct_model(model_data)
        result = model.perform_interaction_effects_analysis(target_variable)
        return convert_to_serializable(result)
    except Exception as e:
        logger.error(f"Error in perform_interaction_effects_analysis: {str(e)}")
        logger.exception("Exception details:")
        raise self.retry(exc=e)

@app.task(bind=True, max_retries=3, default_retry_delay=300)
def perform_counterfactual_analysis(self, model_data, data_dict, interventions, target_variable):
    try:
        model = reconstruct_model(model_data)
        data = pd.DataFrame(data_dict)
        return convert_to_serializable(model.perform_counterfactual_analysis(interventions, target_variable))
    except Exception as e:
        logger.error(f"Error in perform_counterfactual_analysis: {str(e)}")
        logger.exception("Exception details:")
        raise self.retry(exc=e)

@app.task(bind=True, max_retries=3, default_retry_delay=300)
def perform_sensitivity_analysis(self, model_data, target_variable):
    try:
        model = reconstruct_model(model_data)
        return convert_to_serializable(model.perform_sensitivity_analysis(target_variable))
    except Exception as e:
        logger.error(f"Error in perform_sensitivity_analysis: {str(e)}")
        logger.exception("Exception details:")
        raise self.retry(exc=e)

@app.task(bind=True, max_retries=3, default_retry_delay=300)
def fit_nonlinear_model(self, model_data, data_dict, target_variable):
    try:
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
    except Exception as e:
        logger.error(f"Error in fit_nonlinear_model: {str(e)}")
        logger.exception("Exception details:")
        raise self.retry(exc=e)

@app.task(bind=True, max_retries=3, default_retry_delay=300)
def bayesian_model_comparison(self, model_data, data_dict, model_datas):
    try:
        model = reconstruct_model(model_data)
        data = pd.DataFrame(data_dict)
        return convert_to_serializable(model.bayesian_model_comparison(data, model_datas))
    except Exception as e:
        logger.error(f"Error in bayesian_model_comparison: {str(e)}")
        logger.exception("Exception details:")
        raise self.retry(exc=e)

@app.task(bind=True, max_retries=3, default_retry_delay=300)
def analyze_age_related_changes(self, model_data, data_dict, age_column, cognitive_measures):
    try:
        model = reconstruct_model(model_data)
        data = pd.DataFrame(data_dict)
        return convert_to_serializable(model.analyze_age_related_changes(age_column, cognitive_measures))
    except Exception as e:
        logger.error(f"Error in analyze_age_related_changes: {str(e)}")
        logger.exception("Exception details:")
        raise self.retry(exc=e)

@app.task(bind=True, max_retries=3, default_retry_delay=300)
def compare_performance(self, model_data, data_dict, target_variable):
    try:
        model = reconstruct_model(model_data)
        data = pd.DataFrame(data_dict)
        return convert_to_serializable(model.compare_performance(model, data, target_variable))
    except Exception as e:
        logger.error(f"Error in compare_performance: {str(e)}")
        logger.exception("Exception details:")
        raise self.retry(exc=e)

@app.task(bind=True, max_retries=3, default_retry_delay=300)
def fit_simple(self, model_data, data_dict, target_variable):
    try:
        model = reconstruct_model(model_data)
        data = pd.DataFrame(data_dict)
        model.fit_simple(data, target_variable)

        return convert_to_serializable({
            'samples': model.samples,
            'data': model.data.to_dict(),
            'categorical_columns': model.categorical_columns,
            'target_variable': model.target_variable,
            'prior_edges': model.prior_edges,
            'num_features': model.num_features
        })
    except Exception as e:
        logger.error(f"Error in fit_simple: {str(e)}")
        logger.exception("Exception details:")
        raise self.retry(exc=e)
