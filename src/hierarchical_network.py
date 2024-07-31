import numpy as np
import pandas as pd
from scipy import stats
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from jax import random, vmap
from functools import partial
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class HierarchicalBayesianNetwork:
    def __init__(self, num_features, max_parents=6, iterations=2000, categorical_columns=None, target_variable=None, prior_edges=None):
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

    def model(self, X, y=None):
        num_samples, num_features = X.shape
        
        with numpyro.plate('features', num_features):
            alpha = numpyro.sample('alpha', dist.Normal(0, 10))
            beta = numpyro.sample('beta', dist.Normal(0, 10))
        
        sigma = numpyro.sample('sigma', dist.HalfNormal(1))
        
        # Use continuous edge weights
        with numpyro.plate('edges', len(self.prior_edges)):
            edge_weights = numpyro.sample('edge_weights', dist.Beta(1, 1))
        
        with numpyro.plate('data', num_samples):
            y_hat = jnp.zeros(num_samples)
            for i, (parent, child) in enumerate(self.prior_edges):
                if parent != self.target_variable and child != self.target_variable:
                    parent_idx = self.data.columns.get_loc(parent)
                    child_idx = self.data.columns.get_loc(child)
                    y_hat += edge_weights[i] * (alpha[parent_idx] * X[:, parent_idx] + beta[child_idx] * X[:, child_idx])
            
            numpyro.sample('y', dist.Normal(y_hat, sigma), obs=y)

    def fit(self, data, prior_edges=None):
        if self.target_variable is None:
            raise ValueError("Target variable is not set.")
        if self.target_variable not in data.columns:
            raise KeyError(f"Target variable '{self.target_variable}' not found in data columns.")

        if prior_edges is not None:
            self.prior_edges = prior_edges
        self.data = data
        self.num_features = len(data.columns) - 1

        X = data.drop(columns=[self.target_variable]).values
        y = data[self.target_variable].values

        kernel = NUTS(self.model)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=self.iterations)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, X=X, y=y)
        self.samples = mcmc.get_samples()
        self.edge_weights = self.samples['edge_weights'].mean(axis=0)

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
        if self.edge_weights is None:
            raise ValueError("Model has not been fitted yet or fitting failed. Call `fit` before computing edge probabilities.")

        edge_weights = self.samples['edge_weights'].mean(axis=0)
        edge_weights_std = self.samples['edge_weights'].std(axis=0)
        
        results = {}
        for i, (parent, child) in enumerate(self.prior_edges):
            if parent != self.target_variable and child != self.target_variable:
                weight = edge_weights[i]
                std = edge_weights_std[i]
                
                # Calculate 95% credible interval
                lower_ci = max(0, weight - 1.96 * std)
                upper_ci = min(1, weight + 1.96 * std)
                
                # Calculate probability that edge is important (weight > 0.5)
                important_prob = (self.samples['edge_weights'][:, i] > 0.5).mean()
                
                results[f"{parent}->{child}"] = {
                    "weight": weight,
                    "std_dev": std,
                    "95%_CI": (lower_ci, upper_ci),
                    "P(important)": important_prob
                }

        return results
    
    def explain_structure_extended(self):
        if self.samples is None:
            return "The network structure has not been learned yet."

        summary = []

        sensitivities = self.compute_sensitivity(self.target_variable)
        hub_threshold = np.percentile(list(sensitivities.values()), 80)
        hubs = [node for node, sensitivity in sensitivities.items() if sensitivity > hub_threshold]
        summary.append(f"Key hub variables: {', '.join(hubs)}")

        edge_probs = self.compute_all_edge_probabilities()
        # Sort by the 'weight' value in each dictionary
        strongest_edges = sorted(edge_probs.items(), key=lambda x: x[1]['weight'], reverse=True)[:5]
        summary.append("Strongest relationships in the network:")
        for edge, stats in strongest_edges:
            summary.append(f"  {edge}: weight = {stats['weight']:.2f}, P(important) = {stats['P(important)']:.2f}")

        return "\n".join(summary)

    def get_key_relationships(self):
        edge_probs = self.compute_all_edge_probabilities()
        return {edge: stats for edge, stats in edge_probs.items() if stats['P(important)'] > 0.5}

    def compute_sensitivity(self, target_variable):
        if self.samples is None:
            raise ValueError("Model has not been fitted yet. Call `fit` before computing sensitivity.")

        sensitivities = {}
        for column in self.data.columns:
            if column != target_variable:
                sensitivities[column] = self.compute_mutual_information_bayesian(column, target_variable)
        return sensitivities

    def compute_mutual_information_bayesian(self, node1, node2, num_bins=10, categorical_columns=[]):
        x = self.data[node1].values
        y = self.data[node2].values

        def mi_model(x_bins, y_bins):
            concentration = jnp.ones((num_bins, num_bins))
            pxy = numpyro.sample('pxy', dist.Dirichlet(concentration))
            px = numpyro.sample('px', dist.Dirichlet(jnp.ones(num_bins)))
            py = numpyro.sample('py', dist.Dirichlet(jnp.ones(num_bins)))
            
            numpyro.sample('x', dist.Categorical(probs=px), obs=x_bins)
            numpyro.sample('y', dist.Categorical(probs=py), obs=y_bins)
        
        if node1 in categorical_columns:
            x_bins = x.astype(int)
        else:
            x_bins = np.digitize(x, np.linspace(x.min(), x.max(), num_bins + 1)[1:-1]) - 1

        if node2 in categorical_columns:
            y_bins = y.astype(int)
        else:
            y_bins = np.digitize(y, np.linspace(y.min(), y.max(), num_bins + 1)[1:-1]) - 1

        kernel = NUTS(mi_model)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
        mcmc.run(random.PRNGKey(0), x_bins, y_bins)
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
            self.fit(train_data, prior_edges=self.prior_edges)  # Pass prior_edges here
            log_likelihood = self.compute_log_likelihood(test_data)
            log_likelihoods.append(log_likelihood)

        return np.mean(log_likelihoods), np.std(log_likelihoods)

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
    
    def analyze_brain_cognitive_correlations(data, brain_features, cognitive_features):
        brain_data = data[brain_features]
        cognitive_data = data[cognitive_features]
        return brain_data.corrwith(cognitive_data, method='pearson').dropna()

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

    def latent_factor_analysis(self, observed_variables):
        def factor_model(X):
            num_samples, num_vars = X.shape
            num_factors = min(num_vars // 2, 5)  # Rule of thumb for number of factors
            
            factor_loadings = numpyro.sample('factor_loadings', 
                                             dist.Normal(0, 1).expand([num_vars, num_factors]))
            factors = numpyro.sample('factors', 
                                     dist.Normal(0, 1).expand([num_samples, num_factors]))
            
            sigma = numpyro.sample('sigma', dist.HalfNormal(1).expand([num_vars]))
            
            mean = jnp.dot(factors, factor_loadings.T)
            numpyro.sample('X', dist.Normal(mean, sigma), obs=X)

        X = self.data[observed_variables].values
        kernel = NUTS(factor_model)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
        mcmc.run(random.PRNGKey(0), X)
        samples = mcmc.get_samples()

        return {
            'factor_loadings': samples['factor_loadings'].mean(axis=0),
            'factors': samples['factors'].mean(axis=0)
        }

    def structural_equation_model(self, observed_variables, latent_variables, path_model):
        def sem_model(X):
            num_samples = X.shape[0]
            num_latent = len(latent_variables)
            num_observed = len(observed_variables)
            
            # Latent variables
            latent = numpyro.sample('latent', 
                                    dist.Normal(0, 1).expand([num_samples, num_latent]))
            
            # Path coefficients
            path_coeff = {path: numpyro.sample(f'path_{path[0]}_{path[1]}', dist.Normal(0, 1))
                          for path in path_model}
            
            # Compute expected values for observed variables
            expected = jnp.zeros((num_samples, num_observed))
            for i, var in enumerate(observed_variables):
                for path in path_model:
                    if path[1] == var:
                        if path[0] in latent_variables:
                            expected = expected.at[:, i].add(
                                latent[:, latent_variables.index(path[0])] * path_coeff[path]
                            )
                        else:
                            expected = expected.at[:, i].add(
                                X[:, observed_variables.index(path[0])] * path_coeff[path]
                            )
            
            # Observe data
            sigma = numpyro.sample('sigma', dist.HalfNormal(1).expand([num_observed]))
            numpyro.sample('X', dist.Normal(expected, sigma), obs=X)

        X = self.data[observed_variables].values
        kernel = NUTS(sem_model)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
        mcmc.run(random.PRNGKey(0), X)
        samples = mcmc.get_samples()

        return {path: samples[f'path_{path[0]}_{path[1]}'].mean() for path in path_model}

    def nonlinear_cognitive_model(self, X, y=None):
        num_samples, num_features = X.shape
        
        # Non-linear transformation of features
        W1 = numpyro.sample('W1', dist.Normal(0, 1).expand([num_features, 10]))
        b1 = numpyro.sample('b1', dist.Normal(0, 1).expand([10]))
        W2 = numpyro.sample('W2', dist.Normal(0, 1).expand([10, 1]))
        b2 = numpyro.sample('b2', dist.Normal(0, 1))
        
        # Non-linear activation function (e.g., ReLU)
        hidden = jnp.maximum(0, jnp.dot(X, W1) + b1)
        y_hat = jnp.dot(hidden, W2) + b2
        
        sigma = numpyro.sample('sigma', dist.HalfNormal(1))
        numpyro.sample('y', dist.Normal(y_hat, sigma), obs=y)

    def fit_nonlinear(self, data, target_variable):
        X = data.drop(columns=[target_variable]).values
        y = data[target_variable].values

        kernel = NUTS(self.nonlinear_cognitive_model)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
        mcmc.run(random.PRNGKey(0), X, y)
        self.samples = mcmc.get_samples()

    def bayesian_model_comparison(self, data, models):
        def compute_waic(model, data):
            kernel = NUTS(model)
            mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
            mcmc.run(random.PRNGKey(0), data)
            samples = mcmc.get_samples()
            log_likelihood = vmap(lambda params: model(params, data))(samples)
            return -2 * (log_likelihood.mean() - log_likelihood.var())

        return {model_name: compute_waic(model, data) for model_name, model in models.items()}

    def compare_performance(model, data, target_variable):
        X = data.drop(columns=[target_variable])
        y = data[target_variable]
        
        # Your model's performance
        y_pred_model = model.predict(data)
        mse_model = mean_squared_error(y, y_pred_model)
        r2_model = r2_score(y, y_pred_model)
        
        # Linear regression performance
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred_lr = lr.predict(X)
        mse_lr = mean_squared_error(y, y_pred_lr)
        r2_lr = r2_score(y, y_pred_lr)
        
        return {
            "NeuroBayesianModel": {"MSE": mse_model, "R2": r2_model},
            "Linear Regression": {"MSE": mse_lr, "R2": r2_lr}
        }

    def get_practical_implications(model, target_variable, feature_thresholds):
        implications = []
        sensitivity = model.compute_sensitivity(target_variable)

        for feature, threshold in feature_thresholds.items():
            if sensitivity[feature] > threshold:
                if feature.startswith("FS_"):
                    implications.append(f"Focus on exercises that promote {feature} preservation or enhancement.")
                elif feature.startswith("NEOFAC_"):
                    trait = feature.split("_")[1]
                    implications.append(f"Encourage {trait} as part of the cognitive training regimen.")

        return implications

    def perform_age_stratified_analysis(data, age_column, target_variable, age_groups):
        results = {}
        for group, (min_age, max_age) in age_groups.items():
            group_data = data[(data[age_column] >= min_age) & (data[age_column] <= max_age)]
            results[group] = {
                "correlation": group_data.corr()[target_variable].to_dict(),
                "mean": group_data.mean().to_dict(),
                "std": group_data.std().to_dict(),
            }
        return results

    def get_clinical_insights(model, target_variable, feature_categories):
        insights = []
        sensitivity = model.compute_sensitivity(target_variable)

        for category, features in feature_categories.items():
            category_features = [f for f in sensitivity if f in features]
            if category_features:
                top_feature = max(category_features, key=lambda x: abs(sensitivity[x]))
                value = sensitivity[top_feature]
                insights.append(f"The most influential {category} feature for {target_variable} is {top_feature} (sensitivity: {value:.2f}). ")

                if category == "Brain Structure":
                    insights[-1] += f"This suggests that {'increases' if value > 0 else 'decreases'} in {top_feature} are associated with {'higher' if value > 0 else 'lower'} {target_variable}."
                elif category == "Personality":
                    insights[-1] += f"This indicates that the personality trait of {top_feature.split('_')[1]} plays a significant role in {target_variable}."

        return insights

    def generate_comprehensive_insights(model, data, target_variable):
        insights = []

        for column in data.columns:
            if column != target_variable:
                correlation = np.corrcoef(data[column], data[target_variable])[0, 1]
                mutual_info = model.compute_mutual_information_bayesian(column, target_variable)
                if abs(correlation) < 0.1 and mutual_info > 0.1:
                    insights.append(f"Potential non-linear relationship between {column} and {target_variable}")

        for col1 in data.columns:
            for col2 in data.columns:
                if col1 != col2 and col1 != target_variable and col2 != target_variable:
                    interaction = data[col1] * data[col2]
                    int_corr = np.corrcoef(interaction, data[target_variable])[0, 1]
                    if abs(int_corr) > 0.2:
                        insights.append(f"Potential interaction effect between {col1} and {col2} on {target_variable}")

        return insights
