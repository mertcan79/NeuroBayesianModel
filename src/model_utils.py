import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from scipy import stats

def simple_linear_model(X, target_variable=None):
    beta = numpyro.sample("beta", dist.Normal(0, 1).expand([X.shape[1]]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    mean = jnp.dot(X, beta)
    numpyro.sample(target_variable, dist.Normal(mean, sigma), obs=None)

def nonlinear_cognitive_model(X, target_variable=None):
    num_samples, num_features = X.shape

    # Reduce number of hidden units
    W1 = numpyro.sample("W1", dist.Normal(0, 1).expand([num_features, 5]))  # Changed from 10 to 5
    b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([5]))
    W2 = numpyro.sample("W2", dist.Normal(0, 1).expand([5, 1]))
    b2 = numpyro.sample("b2", dist.Normal(0, 1))

    hidden = jnp.maximum(0, jnp.dot(X, W1) + b1)
    y_hat = jnp.dot(hidden, W2) + b2

    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    numpyro.sample(target_variable, dist.Normal(y_hat, sigma), obs=None)

def perform_sensitivity_analysis(model, data, target_variable, features, num_samples=1000):
    """
    Perform sensitivity analysis to assess the impact of features on the target variable.

    Args:
        model: Fitted model object.
        data (pd.DataFrame): Input data.
        target_variable (str): Name of the target variable.
        features (list): List of features to analyze.
        num_samples (int): Number of samples for Monte Carlo simulation.

    Returns:
        pd.DataFrame: Sensitivity indices for each feature.
    """
    sensitivity_indices = {}

    for feature in features:
        feature_data = data[feature]
        perturbed_data = data.copy()

        original_predictions = model.predict(data)[target_variable]

        perturbed_predictions = []
        for _ in range(num_samples):
            perturbed_data[feature] = np.random.permutation(feature_data)
            perturbed_predictions.append(model.predict(perturbed_data)[target_variable])

        perturbed_predictions = np.array(perturbed_predictions)

        sensitivity_index = 1 - (np.var(perturbed_predictions.mean(axis=0)) / np.var(original_predictions))
        sensitivity_indices[feature] = sensitivity_index

    return pd.DataFrame(sensitivity_indices, index=['Sensitivity Index']).T.sort_values('Sensitivity Index', ascending=False)

def perform_counterfactual_analysis(model, data, target_variable, interventions, num_samples=1000):
    """
    Perform counterfactual analysis to assess the impact of interventions on the target variable.

    Args:
        model: Fitted model object.
        data (pd.DataFrame): Input data.
        target_variable (str): Name of the target variable.
        interventions (dict): Dictionary of interventions to apply.
        num_samples (int): Number of samples for uncertainty estimation.

    Returns:
        dict: Counterfactual effects for each intervention.
    """
    original_predictions = model.predict(data)[target_variable]
    counterfactual_effects = {}

    for feature, value in interventions.items():
        counterfactual_data = data.copy()
        counterfactual_data[feature] = value

        counterfactual_predictions = []
        for _ in range(num_samples):
            sample = model.predict(counterfactual_data)[target_variable]
            counterfactual_predictions.append(sample)

        counterfactual_predictions = np.array(counterfactual_predictions)

        effect = counterfactual_predictions.mean() - original_predictions.mean()
        ci_lower, ci_upper = np.percentile(counterfactual_predictions - original_predictions, [2.5, 97.5])

        counterfactual_effects[feature] = {
            'mean_effect': effect,
            '95%_CI': (ci_lower, ci_upper)
        }

    return counterfactual_effects