import jax.numpy as jnp
import jax
from jax import random, vmap, lax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from functools import partial
import jax.scipy.stats as jstats
from typing import Dict, Tuple, Any


class ImprovedDynamicBayesianNetwork:
    def __init__(self, num_features: int, num_states: int = 2, max_parents: int = 2):
        self.num_features = num_features
        self.num_states = num_states
        self.max_parents = max_parents
        self.samples = None
        self.structure = None

    def discretize_mean_std(self, data: jnp.ndarray, cl: float = -0.5, ch: float = 1.5) -> jnp.ndarray:
        """
        Perform soft discretization using mean and standard deviation.
        """
        mean = jnp.mean(data, axis=0)
        std = jnp.std(data, axis=0)
        low = mean + cl * std
        high = mean + ch * std
        return jnp.clip((data - low) / (high - low), 0, 1)

    def optimal_sample_size(self) -> int:
        """
        Calculate the optimal sample size based on the number of states and max parents.
        """
        K = 3  # ternary discretization
        lambda_val = 20.41
        return int(lambda_val * (K ** (self.max_parents + 1)))

    def hill_climbing(self, data: jnp.ndarray, max_iterations: int = 100) -> jnp.ndarray:
        """
        Perform hill climbing to learn the network structure.
        """
        def body_fun(state):
            i, j, current_structure, current_score, improved = state
            new_structure = current_structure.at[i, j].set(~current_structure[i, j])
            new_score = self.score_structure(data, new_structure)
            update_cond = (jnp.sum(new_structure[i]) <= self.max_parents) & (new_score > current_score)
            return (
                (i + 1) % self.num_states,
                (j + 1) % self.num_states,
                jnp.where(update_cond, new_structure, current_structure),
                jnp.where(update_cond, new_score, current_score),
                improved | update_cond
            )

        def cond_fun(state):
            _, _, _, _, improved = state
            return improved

        initial_structure = jnp.zeros((self.num_states, self.num_states), dtype=bool)
        initial_score = self.score_structure(data, initial_structure)
        initial_state = (0, 0, initial_structure, initial_score, True)

        final_state = lax.while_loop(cond_fun, body_fun, initial_state)
        return final_state[2]  # Return the final structure

    def score_structure(self, data: jnp.ndarray, structure: jnp.ndarray) -> float:
        """
        Compute the score for a given network structure.
        """
        num_timesteps, _ = data.shape
        log_likelihood = self.compute_log_likelihood(data, structure)
        num_params = jnp.sum(structure) * (self.num_states - 1)
        return log_likelihood - 0.5 * num_params * jnp.log(num_timesteps)

    def compute_log_likelihood(self, data: jnp.ndarray, structure: jnp.ndarray) -> float:
        """
        Compute the log-likelihood of the data given the structure.
        """
        num_timesteps, _ = data.shape

        initial_probs = jnp.ones(self.num_states) / self.num_states
        trans_matrix = jnp.where(structure, 1.0, 0.0)
        trans_matrix /= trans_matrix.sum(axis=1, keepdims=True)

        means = jnp.zeros((self.num_states, self.num_features))
        scales = jnp.ones((self.num_states, self.num_features))

        def forward_step(carry, obs):
            alpha = carry
            emission_probs = jnp.prod(jstats.norm.pdf(obs, means, scales), axis=1)
            alpha = jnp.dot(alpha, trans_matrix) * emission_probs
            alpha /= alpha.sum()
            return alpha, alpha

        initial_alpha = initial_probs * jnp.prod(jstats.norm.pdf(data[0], means, scales), axis=1)
        initial_alpha /= initial_alpha.sum()

        _, alphas = jax.lax.scan(forward_step, initial_alpha, data[1:])

        return jnp.sum(jnp.log(jnp.sum(alphas, axis=1)))

    def model(self, data: jnp.ndarray, structure: jnp.ndarray):
        """
        Define the probabilistic model for the DBN.
        """
        num_timesteps, _ = data.shape

        initial_probs = numpyro.sample('initial_probs', dist.Dirichlet(jnp.ones(self.num_states)))
        
        with numpyro.plate('states', self.num_states):
            trans_matrix_raw = numpyro.sample('trans_matrix_raw', dist.Dirichlet(jnp.ones(self.num_states)))
        trans_matrix = trans_matrix_raw * structure

        means = numpyro.sample('means', dist.Normal(0, 1).expand([self.num_states, self.num_features]))
        scales = numpyro.sample('scales', dist.HalfNormal(1).expand([self.num_states, self.num_features]))

        def emission(state, t):
            return numpyro.sample(f'obs_{t}', dist.Normal(means[state], scales[state]), obs=data[t])

        state = numpyro.sample('state_0', dist.Categorical(initial_probs))
        emission(state, 0)

        for t in range(1, num_timesteps):
            state = numpyro.sample(f'state_{t}', dist.Categorical(trans_matrix[state]))
            emission(state, t)

    def fit(self, data: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Fit the DBN to the given data.
        """
        min_samples = self.optimal_sample_size()
        if len(data) < min_samples:
            raise ValueError(f"Not enough samples. Need at least {min_samples}")

        discretized_data = self.discretize_mean_std(data)
        self.structure = self.hill_climbing(discretized_data)

        rng_key = random.PRNGKey(0)
        kernel = NUTS(partial(self.model, structure=self.structure))
        mcmc = MCMC(kernel, num_warmup=100, num_samples=200)
        mcmc.run(rng_key, data=data)
        self.samples = mcmc.get_samples()
        return self.samples

    def predict(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        Predict the most likely state sequence for the given data.
        """
        if self.samples is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        discretized_data = self.discretize_mean_std(data)
        trans_matrix = self.samples['trans_matrix'].mean(axis=0) * self.structure
        means = self.samples['means'].mean(axis=0)
        scales = self.samples['scales'].mean(axis=0)
        initial_probs = self.samples['initial_probs'].mean(axis=0)

        num_timesteps = len(discretized_data)
        log_probs = vmap(lambda obs: vmap(lambda mu, scale: dist.Normal(mu, scale).log_prob(obs).sum())(means, scales))(discretized_data)

        def viterbi_step(carry, t):
            prev_logp, prev_path = carry
            curr_logp = prev_logp[:, None] + jnp.log(trans_matrix) + log_probs[t]
            best_logp = jnp.max(curr_logp, axis=0)
            best_path = jnp.argmax(curr_logp, axis=0)
            return (best_logp, best_path), best_path

        init_logp = jnp.log(initial_probs) + log_probs[0]
        (final_logp, _), paths = jax.lax.scan(viterbi_step, (init_logp, jnp.zeros(self.num_states, dtype=jnp.int32)), jnp.arange(1, num_timesteps))

        best_path = jnp.zeros(num_timesteps, dtype=jnp.int32)
        best_path = best_path.at[-1].set(jnp.argmax(final_logp))

        def backtrace(carry, t):
            best_path, paths = carry
            best_path = best_path.at[t].set(paths[t, best_path[t+1]])
            return (best_path, paths), None

        (best_path, _), _ = jax.lax.scan(backtrace, (best_path, paths), jnp.arange(num_timesteps - 2, -1, -1))

        return best_path

    def get_edge_probabilities(self) -> Dict[str, float]:
        """
        Compute the probabilities of edges in the network.
        """
        if self.samples is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        edge_probs = {}
        trans_matrix = self.samples['trans_matrix'].mean(axis=0)
        for i in range(self.num_states):
            for j in range(self.num_states):
                if self.structure[i, j]:
                    edge_probs[f'State_{i}->State_{j}'] = float(trans_matrix[i, j])

        means = self.samples['means'].mean(axis=0)
        for i in range(self.num_states):
            for j in range(self.num_features):
                edge_probs[f'State_{i}->Feature_{j}'] = float(jnp.abs(means[i, j]))

        return edge_probs