import numpy as np
import pandas as pd
from scipy import stats
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from jax import random, vmap
from functools import partial

class HierarchicalBayesianNetwork:
    def __init__(self, num_features, max_parents=6, iterations=2000, categorical_columns=None, target_variable=None):
        self.num_features = num_features
        self.max_parents = max_parents
        self.iterations = iterations
        self.categorical_columns = categorical_columns or []
        self.target_variable = target_variable
        self.prior_edges = None
        self.samples = None
        self.data = None

    def model(self, X, y=None):
        num_samples, num_features = X.shape
        
        alpha = numpyro.sample('alpha', dist.Normal(0, 10).expand([num_features]))
        beta = numpyro.sample('beta', dist.Normal(0, 10).expand([num_features]))
        sigma = numpyro.sample('sigma', dist.HalfNormal(1))
        
        y_hat = jnp.zeros(num_samples)
        for parent, child in self.prior_edges:
            if parent != self.target_variable and child != self.target_variable:
                parent_idx = self.data.columns.get_loc(parent)
                child_idx = self.data.columns.get_loc(child)
                y_hat += alpha[parent_idx] * X[:, parent_idx] + beta[child_idx] * X[:, child_idx]
        
        numpyro.sample('y', dist.Normal(y_hat, sigma), obs=y)

    def fit(self, data, prior_edges=None):
        if self.target_variable is None:
            raise ValueError("Target variable is not set.")
        if self.target_variable not in data.columns:
            raise KeyError(f"Target variable '{self.target_variable}' not found in data columns.")

        self.prior_edges = prior_edges
        self.data = data
        self.num_features = len(data.columns) - 1

        X = data.drop(columns=[self.target_variable]).values
        y = data[self.target_variable].values

        kernel = NUTS(self.model)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=self.iterations)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, X, y)
        self.samples = mcmc.get_samples()

    def predict(self, X):
        if self.target_variable is None:
            raise ValueError("Target variable is not set.")
        if self.samples is None:
            raise ValueError("Model has not been fitted yet. Call `fit` before `predict`.")

        alpha_mean = self.samples['alpha'].mean(axis=0)
        beta_mean = self.samples['beta'].mean(axis=0)

        X_tensor = X.drop(columns=[self.target_variable]).values
        y_hat = np.zeros(X_tensor.shape[0])

        for parent, child in self.prior_edges:
            parent_idx = X.columns.get_loc(parent)
            child_idx = X.columns.get_loc(child)
            y_hat += alpha_mean[parent_idx] * X_tensor[:, parent_idx] + beta_mean[child_idx] * X_tensor[:, child_idx]

        return y_hat

    def compute_edge_probability(self, data):
        if self.samples is None:
            raise ValueError("Model has not been fitted yet. Call `fit` before computing edge probabilities.")

        edge_probs = {}
        for parent, child in self.prior_edges:
            parent_idx = data.columns.get_loc(parent)
            child_idx = data.columns.get_loc(child)
            edge_probs[(parent, child)] = np.abs(np.mean(self.samples['alpha'][:, parent_idx] * self.samples['beta'][:, child_idx]))
        return edge_probs

    def compute_log_likelihood(self, data):
        if self.samples is None:
            raise ValueError("Model has not been fitted yet. Call `fit` before computing log likelihood.")

        y_hat = self.predict(data)
        sigma = self.samples['sigma'].mean()
        return dist.Normal(y_hat, sigma).log_prob(data[self.target_variable]).sum()

    def compute_model_evidence(self, data):
        if self.samples is None:
            raise ValueError("Model has not been fitted yet. Call `fit` before computing model evidence.")

        y_hat = self.predict(data)
        sigma = self.samples['sigma'].mean()
        log_likelihoods = dist.Normal(y_hat, sigma).log_prob(data[self.target_variable])
        return -2 * (np.mean(log_likelihoods) - np.var(log_likelihoods))

    def get_parameter_estimates(self):
        if self.samples is None:
            raise ValueError("Model has not been fitted yet. Call `fit` before getting parameter estimates.")

        return {
            "alpha": self.samples['alpha'].mean(axis=0),
            "beta": self.samples['beta'].mean(axis=0),
            "sigma": self.samples['sigma'].mean(),
        }

    def compute_all_edge_probabilities(self):
        if self.samples is None:
            raise ValueError("Model has not been fitted yet or fitting failed. Call `fit` before computing edge probabilities.")

        edge_probabilities = {}
        for parent, child in self.prior_edges:
            parent_idx = self.data.columns.get_loc(parent)
            child_idx = self.data.columns.get_loc(child)
            prob = np.abs(np.mean(self.samples['alpha'][:, parent_idx] * self.samples['beta'][:, child_idx]))
            edge_probabilities[f"{parent}->{child}"] = prob
        return edge_probabilities

    def explain_structure_extended(self):
        if self.samples is None:
            return "The network structure has not been learned yet."

        summary = []

        sensitivities = self.compute_sensitivity(self.target_variable)
        hub_threshold = np.percentile(list(sensitivities.values()), 80)
        hubs = [node for node, sensitivity in sensitivities.items() if sensitivity > hub_threshold]
        summary.append(f"Key hub variables: {', '.join(hubs)}")

        edge_probs = self.compute_all_edge_probabilities()
        strongest_edges = sorted(edge_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        summary.append("Strongest relationships in the network:")
        for edge, prob in strongest_edges:
            summary.append(f"  {edge}: probability = {prob:.2f}")

        return "\n".join(summary)

    def get_key_relationships(self):
        edge_probs = self.compute_all_edge_probabilities()
        return {edge: prob for edge, prob in edge_probs.items() if prob > 0.5}

    def compute_sensitivity(self, target_variable):
        if self.samples is None:
            raise ValueError("Model has not been fitted yet. Call `fit` before computing sensitivity.")

        sensitivities = {}
        for column in self.data.columns:
            if column != target_variable:
                sensitivities[column] = self.compute_mutual_information_bayesian(column, target_variable)
        return sensitivities

    def compute_mutual_information_bayesian(self, node1, node2, num_bins=10):
        x = self.data[node1].values
        y = self.data[node2].values
        
        def mi_model(x, y):
            # Concentration parameter for Dirichlet distribution
            concentration = jnp.ones(num_bins ** 2)
            
            # Sample from Dirichlet distribution for the joint probabilities
            pxy = numpyro.sample('pxy', dist.Dirichlet(concentration))
            
            # Sample from Dirichlet distribution for the marginal probabilities
            px = numpyro.sample('px', dist.Dirichlet(jnp.ones(num_bins)))
            py = numpyro.sample('py', dist.Dirichlet(jnp.ones(num_bins)))
            
            # Convert continuous data to categorical bins
            x_bins = np.digitize(x, np.linspace(x.min(), x.max(), num_bins + 1)[1:-1])
            y_bins = np.digitize(y, np.linspace(y.min(), y.max(), num_bins + 1)[1:-1])
            
            # Define observed variables
            numpyro.sample('x', dist.Categorical(px), obs=x_bins)
            numpyro.sample('y', dist.Categorical(py), obs=y_bins)
            
        kernel = NUTS(mi_model)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
        mcmc.run(random.PRNGKey(0), x, y)
        samples = mcmc.get_samples()
        
        px = samples['px'].mean(axis=0)
        py = samples['py'].mean(axis=0)
        pxy = samples['pxy'].mean(axis=0)
        
        mi = jnp.sum(pxy * jnp.log(pxy / (px[:, None] * py[None, :])))
        return mi

    def cross_validate_bayesian(self, data, k=5):
        def split_data(data, k):
            n = len(data)
            fold_size = n // k
            indices = np.arange(n)
            np.random.shuffle(indices)
            for i in range(k):
                test_indices = indices[i*fold_size:(i+1)*fold_size]
                train_indices = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
                yield data.iloc[train_indices], data.iloc[test_indices]

        log_likelihoods = []
        for train_data, test_data in split_data(data, k):
            self.fit(train_data)
            log_likelihood = self.compute_log_likelihood(test_data)
            log_likelihoods.append(log_likelihood)

        return np.mean(log_likelihoods)

    def get_target_variable(self):
        return self.target_variable

    def get_clinical_implications(self):
        if self.samples is None:
            raise ValueError("Model has not been fitted yet. Call `fit` before getting clinical implications.")

        implications = {}
        for column in self.data.columns:
            if column in self.categorical_columns:
                implications[column] = {
                    "categories": self.data[column].unique().tolist(),
                    "probabilities": self.data[column].value_counts(normalize=True).to_dict(),
                }
            else:
                implications[column] = {
                    "mean": self.data[column].mean(),
                    "std": self.data[column].std(),
                }
        return implications

    def analyze_age_dependent_relationships(self, age_column, target_variable):
        median_age = self.data[age_column].median()
        young_data = self.data[self.data[age_column] < median_age]
        old_data = self.data[self.data[age_column] >= median_age]

        young_corr = young_data.corr()[target_variable]
        old_corr = old_data.corr()[target_variable]

        age_differences = {}
        for feature in young_corr.index:
            if feature != target_variable:
                age_differences[feature] = old_corr[feature] - young_corr[feature]

        return age_differences

    def perform_interaction_effects_analysis(self, target):
        interactions = {}
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for i in range(len(numeric_columns)):
            for j in range(i + 1, len(numeric_columns)):
                col1, col2 = numeric_columns[i], numeric_columns[j]
                interaction_term = self.data[col1] * self.data[col2]
                correlation = np.corrcoef(interaction_term, self.data[target])[0, 1]
                interactions[f"{col1}_{col2}"] = correlation
        return interactions

    def perform_counterfactual_analysis(self, interventions, target_variable):
        original_prediction = self.predict(self.data)
        counterfactual_data = self.data.copy()
        for variable, value in interventions.items():
            counterfactual_data[variable] = value
        counterfactual_prediction = self.predict(counterfactual_data)
        return {
            "original_prediction": original_prediction.mean(),
            "counterfactual_prediction": counterfactual_prediction.mean(),
            "difference": (counterfactual_prediction - original_prediction).mean(),
        }

    def perform_sensitivity_analysis(self, target, perturbation=0.1):
        results = {}
        for column in self.data.columns:
            if column != target:
                perturbed_data = self.data.copy()
                perturbed_data[column] *= 1 + perturbation
                original_prediction = self.predict(self.data)
                perturbed_prediction = self.predict(perturbed_data)
                sensitivity = np.mean(np.abs(perturbed_prediction - original_prediction))
                results[column] = sensitivity
        return results
    
    def analyze_brain_cognitive_correlations(self, brain_features, cognitive_features):
        brain_data = self.data[brain_features]
        cognitive_data = self.data[cognitive_features]
        return brain_data.corrwith(cognitive_data)

    def analyze_age_related_changes(self, age_column, target_columns):
        results = {}
        for col in target_columns:
            slope, intercept, r_value, p_value, std_err = stats.linregress(self.data[age_column], self.data[col])
            results[col] = {
                "slope": slope,
                "intercept": intercept,
                "r_value": r_value,
                "p_value": p_value,
                "std_err": std_err,
            }
        return results
    
    def analyze_cognitive_trajectories(self, age_column, cognitive_measures):
        def trajectory_model(age, measure):
            intercept = numpyro.sample('intercept', dist.Normal(0, 10))
            slope = numpyro.sample('slope', dist.Normal(0, 1))
            sigma = numpyro.sample('sigma', dist.HalfNormal(1))
            mu = intercept + slope * age
            numpyro.sample('y', dist.Normal(mu, sigma), obs=measure)

        results = {}
        for measure in cognitive_measures:
            kernel = NUTS(trajectory_model)
            mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
            mcmc.run(random.PRNGKey(0), self.data[age_column].values, self.data[measure].values)
            samples = mcmc.get_samples()
            results[measure] = {
                'intercept': samples['intercept'].mean(),
                'slope': samples['slope'].mean(),
                'intercept_std': samples['intercept'].std(),
                'slope_std': samples['slope'].std(),
            }
        return results