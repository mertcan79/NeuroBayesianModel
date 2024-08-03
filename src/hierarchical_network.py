import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
import logging
from numpyro.diagnostics import split_gelman_rubin

from insights import perform_age_stratified_analysis, analyze_brain_cognitive_correlations
from validation import cross_validate_bayesian, compute_mutual_information_bayesian
from model_utils import simple_linear_model, nonlinear_cognitive_model

logger = logging.getLogger(__name__)

class HierarchicalBayesianNetwork:
    def __init__(
        self,
        num_features: int,
        max_parents: int = 2,
        iterations: int = 50,
        categorical_columns: list = None,
        target_variable: str = None,
        prior_edges: list = None,
    ):
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
        self.prior_edges = prior_edges if prior_edges is not None else []
        self.edge_probabilities = None
        self.samples = None
        self.data = None
        self.edge_weights = None

    def model(self, params, X, y=None):
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


    def fit(self, data: pd.DataFrame, prior_edges: list = None):
        """
        Fit the model to the data.

        Args:
            data (pd.DataFrame): Input data.
            prior_edges (list, optional): List of prior edges in the network. Defaults to None.
        """
        if self.target_variable is None:
            raise ValueError("Target variable is not set.")
        if self.target_variable not in data.columns:
            raise KeyError(
                f"Target variable '{self.target_variable}' not found in data columns."
            )

        if prior_edges is not None:
            self.prior_edges = prior_edges
        self.data = data
        self.num_features = len(data.columns) - 1

        X = data.drop(columns=[self.target_variable]).values
        y = data[self.target_variable].values

        # Initialize samples with dummy values
        self.samples = {
            'alpha': jnp.zeros(self.num_features),
            'beta': jnp.zeros(self.num_features),
            'sigma': jnp.array(1.0),
            'edge_weights': jnp.zeros(len(self.prior_edges))
        }

        def model_wrapper(X, y):
            return self.model(self.samples, X, y)

        kernel = NUTS(model_wrapper)
        mcmc = MCMC(kernel, num_warmup=100, num_samples=self.iterations)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, X=X, y=y)
        self.samples = mcmc.get_samples()
        self.edge_weights = self.samples["edge_weights"].mean(axis=0)

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
            rng_key = random.PRNGKey(0)
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

    def compute_all_edge_probabilities(self):
        """
        Compute the probabilities for all edges.

        Returns:
            dict: Dictionary containing the edge probabilities.
        """
        if self.edge_weights is None:
            raise ValueError(
                "Model has not been fitted yet or fitting failed. Call `fit` before computing edge probabilities."
            )

        edge_weights = self.samples["edge_weights"].mean(axis=0)
        edge_weights_std = self.samples["edge_weights"].std(axis=0)

        results = {}
        for i, (parent, child) in enumerate(self.prior_edges):
            if parent != self.target_variable and child != self.target_variable:
                weight = np.array(edge_weights[i]).item()  # Convert to Python scalar
                std = np.array(edge_weights_std[i]).item()  # Convert to Python scalar

                # Calculate 95% credible interval
                lower_ci = max(0, weight - 1.96 * std)
                upper_ci = min(1, weight + 1.96 * std)

                # Calculate probability that edge is important (weight > 0.5)
                important_prob = np.array((self.samples["edge_weights"][:, i] > 0.5).mean()).item()

                results[f"{parent}->{child}"] = {
                    "weight": weight,
                    "std_dev": std,
                    "95%_CI": (lower_ci, upper_ci),
                    "P(important)": important_prob,
                }

        return results

    def explain_structure_extended(self):
        """
        Explain the structure of the network.

        Returns:
            str: Explanation of the network structure.
        """
        if self.samples is None:
            return "The network structure has not been learned yet."

        summary = []

        sensitivities = self.compute_sensitivity(self.target_variable)
        hub_threshold = np.percentile(list(sensitivities.values()), 80)
        hubs = [
            node
            for node, sensitivity in sensitivities.items()
            if sensitivity > hub_threshold
        ]
        summary.append(f"Key hub variables: {', '.join(hubs)}")

        edge_probs = self.compute_all_edge_probabilities()
        # Sort by the 'weight' value in each dictionary
        strongest_edges = sorted(
            edge_probs.items(), key=lambda x: x[1]["weight"], reverse=True
        )[:5]
        summary.append("Strongest relationships in the network:")
        for edge, stats in strongest_edges:
            summary.append(
                f"  {edge}: weight = {stats['weight']:.2f}, P(important) = {stats['P(important)']:.2f}"
            )

        return "\n".join(summary)

    def get_key_relationships(self):
        """
        Get the key relationships in the network.

        Returns:
            dict: Dictionary containing the key relationships.
        """
        edge_probs = self.compute_all_edge_probabilities()
        return {
            edge: stats
            for edge, stats in edge_probs.items()
            if stats["P(important)"] > 0.5
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

    def perform_age_stratified_analysis(self, age_column, target_variable, age_groups):
        """
        Perform age-stratified analysis on the data.

        Args:
            age_column (str): Column containing age information.
            target_variable (str): The target variable to analyze.
            age_groups (dict): Dictionary of age groups and their corresponding ranges.

        Returns:
            dict: Dictionary containing the results of the analysis.
        """
        return perform_age_stratified_analysis(self.data, age_column, target_variable, age_groups)

    def analyze_brain_cognitive_correlations(self, brain_features, cognitive_features):
        """
        Analyze the correlations between brain features and cognitive features.

        Args:
            brain_features (list): List of brain features.
            cognitive_features (list): List of cognitive features.

        Returns:
            pd.Series: Series containing the correlations between brain features and cognitive features.
        """
        return analyze_brain_cognitive_correlations(self.data, brain_features, cognitive_features)

    def cross_validate_bayesian(self, data, k=5):
        """
        Perform k-fold cross-validation using the Bayesian model.

        Args:
            data (pd.DataFrame): Input data.
            k (int, optional): Number of folds for cross-validation. Defaults to 5.

        Returns:
            tuple: Mean and standard deviation of the log-likelihoods.
        """
        return cross_validate_bayesian(self, data, k)

    def compute_mutual_information_bayesian(self, node1, node2, num_bins=10, categorical_columns=[]):
        """
        Compute the mutual information between two nodes using the Bayesian model.

        Args:
            node1 (str): First node.
            node2 (str): Second node.
            num_bins (int, optional): Number of bins for discretizing continuous variables. Defaults to 10.
            categorical_columns (list, optional): List of categorical columns in the data. Defaults to an empty list.

        Returns:
            float: Mutual information between the two nodes.
        """
        return compute_mutual_information_bayesian(self.data, node1, node2, num_bins, categorical_columns)

    def fit_simple(self, data, target_variable):
        """
        Fit a simple linear model to the data.

        Args:
            data (pd.DataFrame): Input data.
            target_variable (str): The target variable to predict.
        """
        X = data.drop(columns=[target_variable]).values
        y = data[target_variable].values

        kernel = NUTS(simple_linear_model)
        mcmc = MCMC(kernel, num_warmup=100, num_samples=200)
        mcmc.run(random.PRNGKey(0), X, y)
        self.samples = mcmc.get_samples()

    def fit_nonlinear(self, data, target_variable):
        """
        Fit a nonlinear cognitive model to the data.

        Args:
            data (pd.DataFrame): Input data.
            target_variable (str): The target variable to predict.
        """
        X = data.drop(columns=[target_variable]).values
        y = data[target_variable].values

        kernel = NUTS(nonlinear_cognitive_model)
        mcmc = MCMC(kernel, num_warmup=50, num_samples=100)

        for i in range(10):  # Check convergence every 50 samples
            mcmc.run(random.PRNGKey(i), X, y, num_samples=50)
            samples = mcmc.get_samples()
            r_hat = split_gelman_rubin(samples)
            if all(r < 1.1 for r in r_hat.values()):
                break

        self.samples = mcmc.get_samples()
