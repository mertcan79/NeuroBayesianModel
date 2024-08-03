import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


logger = logging.getLogger(__name__)

def cross_validate_bayesian(model, data, k=5):
    """
    Perform k-fold cross-validation using the Bayesian model.

    Args:
        model (HierarchicalBayesianNetwork): The Bayesian model to use for cross-validation.
        data (pd.DataFrame): Input data.
        k (int, optional): Number of folds for cross-validation. Defaults to 5.

    Returns:
        tuple: Mean and standard deviation of the log-likelihoods.
    """
    def split_data(data, k):
        n = len(data)
        fold_size = n // k
        indices = np.arange(n)
        np.random.shuffle(indices)
        for i in range(k):
            test_indices = indices[i * fold_size : (i + 1) * fold_size]
            train_indices = np.concatenate(
                [indices[: i * fold_size], indices[(i + 1) * fold_size :]]
            )
            yield data.iloc[train_indices], data.iloc[test_indices]

    log_likelihoods = []
    for i, (train_data, test_data) in enumerate(split_data(data, k)):
        logger.info(f"Starting fold {i+1}/{k}")
        try:
            model.fit(train_data, prior_edges=model.prior_edges)
            log_likelihood = model.compute_log_likelihood(test_data)
            log_likelihoods.append(log_likelihood)
            logger.info(f"Fold {i+1} completed. Log-likelihood: {log_likelihood}")
        except Exception as e:
            logger.error(f"Error in fold {i+1}: {str(e)}")
            logger.exception("Exception details:")

    if not log_likelihoods:
        raise ValueError("No successful folds in cross-validation")

    return float(np.mean(log_likelihoods)), float(np.std(log_likelihoods))

def compute_mutual_information_bayesian(data, node1, node2, num_bins=10, categorical_columns=[]):
    """
    Compute the mutual information between two nodes using the Bayesian model.

    Args:
        data (pd.DataFrame): Input data.
        node1 (str): First node.
        node2 (str): Second node.
        num_bins (int, optional): Number of bins for discretizing continuous variables. Defaults to 10.
        categorical_columns (list, optional): List of categorical columns in the data. Defaults to an empty list.

    Returns:
        float: Mutual information between the two nodes.
    """
    x = data[node1].values
    y = data[node2].values

    def mi_model(x_bins, y_bins):
        concentration = jnp.ones((num_bins, num_bins))
        pxy = numpyro.sample("pxy", dist.Dirichlet(concentration))
        px = numpyro.sample("px", dist.Dirichlet(jnp.ones(num_bins)))
        py = numpyro.sample("py", dist.Dirichlet(jnp.ones(num_bins)))

        numpyro.sample("x", dist.Categorical(probs=px), obs=x_bins)
        numpyro.sample("y", dist.Categorical(probs=py), obs=y_bins)

    if node1 in categorical_columns:
        x_bins = x.astype(int)
    else:
        x_bins = (
            np.digitize(x, np.linspace(x.min(), x.max(), num_bins + 1)[1:-1]) - 1
        )

    if node2 in categorical_columns:
        y_bins = y.astype(int)
    else:
        y_bins = (
            np.digitize(y, np.linspace(y.min(), y.max(), num_bins + 1)[1:-1]) - 1
        )

    kernel = NUTS(mi_model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
    mcmc.run(random.PRNGKey(0), x_bins, y_bins)
    samples = mcmc.get_samples()

    px = samples["px"].mean(axis=0)
    py = samples["py"].mean(axis=0)
    pxy = samples["pxy"].mean(axis=0)

    mi = jnp.sum(pxy * jnp.log(pxy / (px[:, None] * py[None, :])))
    return mi

def compare_performance(models, data, target_variable, test_size=0.2, random_state=42):
    """
    Compare the performance of multiple models.

    Args:
        models (dict): Dictionary of model objects.
        data (pd.DataFrame): Input data.
        target_variable (str): Name of the target variable.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random state for reproducibility.

    Returns:
        pd.DataFrame: Performance metrics for each model.
    """
    X = data.drop(columns=[target_variable])
    y = data[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }

    return pd.DataFrame(results).T

def bayesian_model_comparison(models, data, target_variable):
    """
    Perform Bayesian model comparison.

    Args:
        models (dict): Dictionary of fitted Bayesian model objects.
        data (pd.DataFrame): Input data.
        target_variable (str): Name of the target variable.

    Returns:
        pd.DataFrame: Model comparison metrics.
    """
    from scipy.special import logsumexp

    results = {}
    for name, model in models.items():
        log_likelihood = model.compute_log_likelihood(data)
        num_params = len(model.get_parameter_estimates())
        bic = -2 * log_likelihood + num_params * np.log(len(data))
        
        results[name] = {
            'Log Likelihood': log_likelihood,
            'Num Parameters': num_params,
            'BIC': bic
        }

    df_results = pd.DataFrame(results).T
    
    # Calculate model probabilities
    bic_values = df_results['BIC'].values
    model_probs = np.exp(-0.5 * (bic_values - np.min(bic_values)))
    model_probs /= np.sum(model_probs)
    df_results['Model Probability'] = model_probs

    return df_results.sort_values('BIC')