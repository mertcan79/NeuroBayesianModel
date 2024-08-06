import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
import logging
from numpyro.diagnostics import split_gelman_rubin
import matplotlib.pyplot as plt

from base import BaseBayesianNetwork

from insights import (
    perform_interaction_effects_analysis
)

from validation import (
    cross_validate_bayesian
)

from model_utils import (
    simple_linear_model,
    perform_sensitivity_analysis
)
logger = logging.getLogger(__name__)

class HierarchicalBayesianNetwork(BaseBayesianNetwork):
    def __init__(self, num_features, max_parents=2, iterations=50, categorical_columns=None, target_variable=None):
        """
        Initialize the HierarchicalBayesianNetwork.

        Args:
            num_features (int): Number of features in the data.
            max_parents (int, optional): Maximum number of parents for each node. Defaults to 2.
            iterations (int, optional): Number of iterations for MCMC sampling. Defaults to 50.
            categorical_columns (list, optional): List of categorical columns in the data. Defaults to None.
            target_variable (str, optional): The target variable to predict. Defaults to None.
            prior_edges (list, optional): List of prior edges in the network. Defaults to None.
        """
        self.num_features = num_features
        self.max_parents = max_parents
        self.iterations = iterations
        self.categorical_columns = categorical_columns or []
        self.target_variable = target_variable
        self.samples = None
        self.edge_weights = None
        self.data = None
        
    def model(self, X, y=None):
        """
        Define the Bayesian model.

        Args:
            params (dict): Dictionary of model parameters.
            X (np.ndarray): Input data.
            y (np.ndarray, optional): Target data. Defaults to None.

        Returns:
            numpyro.sample: Sampled values from the model.
        """
        num_samples, num_features = X.shape
        rng_key = random.PRNGKey(0)

        alpha = numpyro.sample("alpha", dist.Normal(0, 10).expand([num_features]), rng_key=rng_key)
        beta = numpyro.sample("beta", dist.Normal(0, 10).expand([num_features]), rng_key=rng_key)
        sigma = numpyro.sample("sigma", dist.HalfNormal(1), rng_key=rng_key)
        edge_weights = numpyro.sample("edge_weights", dist.Beta(1, 1).expand([len(self.prior_edges)]), rng_key=rng_key)

        with numpyro.plate("data", num_samples):
            y_hat = jnp.zeros(num_samples)
            for i, (parent, child) in enumerate(self.prior_edges):
                if parent != self.target_variable and child != self.target_variable:
                    parent_idx = self.data.columns.get_loc(parent)
                    child_idx = self.data.columns.get_loc(child)
                    y_hat += edge_weights[i] * (
                        alpha[parent_idx] * X[:, parent_idx]
                        + beta[child_idx] * X[:, child_idx]
                    )

            return numpyro.sample("y", dist.Normal(y_hat, sigma), obs=y, rng_key=rng_key)

    def fit(self, data):
        """
        Fit the model to the data.

        Args:
            data (np.ndarray or pd.DataFrame): Input data. If numpy array, last column is assumed to be the target variable.
        """
        if isinstance(data, pd.DataFrame):
            if self.target_variable is None:
                raise ValueError("Target variable is not set.")
            if self.target_variable not in data.columns:
                raise KeyError(f"Target variable '{self.target_variable}' not found in data columns.")
            X = data.drop(columns=[self.target_variable]).values
            y = data[self.target_variable].values
        else:
            X = data[:, :-1]
            y = data[:, -1]

        self.num_features = X.shape[1]

        def model_wrapper(X, y):
            return simple_linear_model(X, target_variable=self.target_variable)

        kernel = NUTS(model_wrapper)
        mcmc = MCMC(kernel, num_warmup=100, num_samples=self.iterations, num_chains=4)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, X=X, y=y)
        self.samples = mcmc.get_samples()

        # Check convergence for each parameter
        r_hat = {}
        for param, samples in self.samples.items():
            if samples.ndim < 2:
                samples = samples.reshape(-1, 1)  # Reshape to 2D if necessary
            if samples.shape[1] < 4:
                print(f"Warning: Insufficient chains for parameter '{param}'. Skipping Gelman-Rubin diagnostic.")
                continue
            r_hat[param] = split_gelman_rubin(samples)

        # Check if 'edge_weights' is present in the samples
        if 'edge_weights' in self.samples:
            self.edge_weights = self.samples["edge_weights"].mean(axis=0)
        else:
            print("Warning: 'edge_weights' not found in samples. Skipping edge weights calculation.")
            self.edge_weights = None

        # Check if 'alpha' is present in the samples
        if 'alpha' in self.samples:
            self.alpha = self.samples["alpha"].mean(axis=0)
        else:
            print("Warning: 'alpha' not found in samples. Skipping alpha calculation.")
            self.alpha = None

        # Check if 'beta' is present in the samples
        if 'beta' in self.samples:
            self.beta = self.samples["beta"].mean(axis=0)
        else:
            print("Warning: 'beta' not found in samples. Skipping beta calculation.")
            self.beta = None

        # Check if 'sigma' is present in the samples
        if 'sigma' in self.samples:
            self.sigma = self.samples["sigma"].mean(axis=0)
        else:
            print("Warning: 'sigma' not found in samples. Skipping sigma calculation.")
            self.sigma = None

        # Check overall convergence
        if not all(r < 1.1 for r in r_hat.values()):
            print("Warning: Model may not have converged.")

        return self.samples

    def predict(self, X):
        """
        Predict the target variable for new data.

        Args:
            X (pd.DataFrame or np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.target_variable is None:
            raise ValueError("Target variable is not set.")
        if self.samples is None:
            raise ValueError(
                "Model has not been fitted yet. Call `fit` before `predict`."
            )

        try:
            alpha_mean = self.samples["alpha"].mean(axis=0)
            beta_mean = self.samples["beta"].mean(axis=0)

            if isinstance(X, pd.DataFrame):
                X_tensor = X.drop(columns=[self.target_variable]).values
            else:
                X_tensor = X

            params = {
                'alpha': alpha_mean,
                'beta': beta_mean,
                'sigma': self.samples['sigma'].mean(),
                'edge_weights': self.edge_weights
            }

            y_hat = self.model(params, X_tensor, None)

            return y_hat
        
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            logger.exception("Exception details:")
            raise

    def compute_edge_probability(self, data):
        """
        Compute the edge probabilities.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            dict: Dictionary containing the edge probabilities.
        """
        if self.samples is None:
            raise ValueError(
                "Model has not been fitted yet. Call `fit` before computing edge probabilities."
            )

        edge_probs = {}
        for parent, child in self.prior_edges:
            parent_idx = data.columns.get_loc(parent)
            child_idx = data.columns.get_loc(child)
            edge_probs[(parent, child)] = np.abs(
                np.mean(
                    self.samples["alpha"][:, parent_idx]
                    * self.samples["beta"][:, child_idx]
                )
            )
        return edge_probs

    def compute_log_likelihood(self, data):
        """
        Compute the log-likelihood of the data.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            float: Log-likelihood of the data.
        """
        if self.samples is None:
            raise ValueError(
                "Model has not been fitted yet. Call `fit` before computing log likelihood."
            )

        try:
            y_hat = self.predict(data)
            sigma = self.samples["sigma"].mean()
            log_prob = dist.Normal(y_hat, sigma).log_prob(jnp.array(data[self.target_variable]))
            return jnp.sum(log_prob)
        except Exception as e:
            logger.error(f"Error in compute_log_likelihood: {str(e)}")
            logger.exception("Exception details:")
            raise

    def compute_model_evidence(self, data):
        """
        Compute the model evidence.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            float: Model evidence.
        """
        if self.samples is None:
            raise ValueError(
                "Model has not been fitted yet. Call `fit` before computing model evidence."
            )

        y_hat = self.predict(data)
        sigma = self.samples["sigma"].mean()
        log_likelihoods = dist.Normal(y_hat, sigma).log_prob(jnp.array(data[self.target_variable]))
        return -2 * (np.mean(log_likelihoods) - np.var(log_likelihoods))

    def compute_edge_probabilities(self):
        if self.samples is None:
            raise ValueError("Model has not been fitted yet. Call `fit` before computing edge probabilities.")
        
        edge_probs = {}
        for i in range(self.num_features):
            for j in range(i+1, self.num_features):
                edge_probs[(i, j)] = np.abs(np.mean(self.samples["alpha"][:, i] * self.samples["beta"][:, j]))
        return edge_probs

    def get_parameter_estimates(self):
        """
        Get the parameter estimates.

        Returns:
            dict: Dictionary containing the parameter estimates.
        """
        if self.samples is None:
            raise ValueError(
                "Model has not been fitted yet. Call `fit` before getting parameter estimates."
            )

        return {
            "alpha": self.samples["alpha"].mean(axis=0),
            "beta": self.samples["beta"].mean(axis=0),
            "sigma": self.samples["sigma"].mean(),
        }

    def compute_sensitivity(self, target_variable):
        """
        Compute the sensitivity of the target variable to each feature.

        Args:
            target_variable (str): The target variable.

        Returns:
            dict: Dictionary containing the sensitivities.
        """
        if self.samples is None:
            raise ValueError(
                "Model has not been fitted yet. Call `fit` before computing sensitivity."
            )

        sensitivities = {}
        for column in self.data.columns:
            if column != target_variable:
                sensitivities[column] = self.compute_mutual_information_bayesian(
                    column, target_variable
                )
        return sensitivities

    def perform_interaction_effects_analysis(self, target_variable, predictors, interaction_terms):
        """
        Analyze interaction effects between predictors on the target variable.

        Args:
            target_variable (str): Name of the target variable.
            predictors (list): List of predictor variables.
            interaction_terms (list): List of tuples representing interaction terms.

        Returns:
            pd.DataFrame: Summary of interaction effects.
        """
        return perform_interaction_effects_analysis(self.data, target_variable, predictors, interaction_terms)

    def perform_sensitivity_analysis(self, target_variable, features, num_samples=1000):
        """
        Perform sensitivity analysis to assess the impact of features on the target variable.

        Args:
            target_variable (str): Name of the target variable.
            features (list): List of features to analyze.
            num_samples (int): Number of samples for Monte Carlo simulation.

        Returns:
            pd.DataFrame: Sensitivity indices for each feature.
        """
        return perform_sensitivity_analysis(self, self.data, target_variable, features, num_samples)

    def perform_cross_validate_bayesian(self, model, data, k=5):
        """
        Perform Bayesian cross validation
        """
        return cross_validate_bayesian(self, model, data, k=5)
        