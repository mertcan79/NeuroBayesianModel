import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
import logging
from numpyro.distributions import MultivariateNormal, Dirichlet, Categorical

from base import BaseBayesianNetwork

logger = logging.getLogger(__name__)

class DynamicBayesianNetwork(BaseBayesianNetwork):
    def __init__(self, num_features, num_states=3, num_timesteps=10, iterations=1000):
        super().__init__()
        self.num_features = num_features
        self.num_states = num_states
        self.num_timesteps = num_timesteps
        self.iterations = iterations
        self.samples = None

    def model(self, data):
        # Initial state distribution
        initial_state_prior = jnp.ones(self.num_states) / self.num_states
        initial_state = numpyro.sample("initial_state", Dirichlet(initial_state_prior))

        # Transition matrix
        transition_matrix = numpyro.sample("transition_matrix", 
            Dirichlet(jnp.ones(self.num_states) / self.num_states).expand([self.num_states]))

        # Emission parameters
        mu = numpyro.sample("mu", MultivariateNormal(jnp.zeros(self.num_features), jnp.eye(self.num_features)).expand([self.num_states]))
        sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0).expand([self.num_states, self.num_features]))

        # Hidden states
        z = numpyro.sample("z", Categorical(initial_state), sample_shape=(self.num_timesteps,))

        # Observations
        with numpyro.plate("timesteps", self.num_timesteps):
            numpyro.sample("obs", MultivariateNormal(mu[z], jnp.diag(sigma[z])), obs=data)

    def fit(self, data):
        """
        Fit the DBN to the data.

        Args:
            data (np.ndarray): Input data with shape (num_timesteps, num_features)
        """
        if data.shape != (self.num_timesteps, self.num_features):
            raise ValueError(f"Data shape {data.shape} does not match expected shape ({self.num_timesteps}, {self.num_features})")

        rng_key = random.PRNGKey(0)
        kernel = NUTS(self.model)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=self.iterations)
        mcmc.run(rng_key, data=data)
        self.samples = mcmc.get_samples()
        return self.samples

    def predict(self, data):
        """
        Predict cognitive states for new data.

        Args:
            data (np.ndarray): Input data with shape (num_timesteps, num_features)

        Returns:
            np.ndarray: Predicted cognitive states
        """
        if self.samples is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Use the mean of the posterior distributions
        transition_matrix = self.samples['transition_matrix'].mean(axis=0)
        mu = self.samples['mu'].mean(axis=0)
        sigma = self.samples['sigma'].mean(axis=0)

        # Viterbi algorithm for state sequence prediction
        log_likelihood = jnp.sum(MultivariateNormal(mu, jnp.diag(sigma)).log_prob(data), axis=1)
        viterbi = jnp.zeros((self.num_timesteps, self.num_states))
        viterbi = viterbi.at[0].set(log_likelihood[0] + jnp.log(self.samples['initial_state'].mean(axis=0)))

        for t in range(1, self.num_timesteps):
            viterbi = viterbi.at[t].set(log_likelihood[t] + jnp.max(viterbi[t-1] + jnp.log(transition_matrix), axis=1))

        # Backtracking
        states = jnp.zeros(self.num_timesteps, dtype=int)
        states = states.at[-1].set(jnp.argmax(viterbi[-1]))
        for t in range(self.num_timesteps - 2, -1, -1):
            states = states.at[t].set(jnp.argmax(viterbi[t] + jnp.log(transition_matrix[:, states[t+1]])))

        return states

    def get_state_probabilities(self, data):
        """
        Get probabilities of each cognitive state for new data.

        Args:
            data (np.ndarray): Input data with shape (num_timesteps, num_features)

        Returns:
            np.ndarray: Probabilities of each cognitive state at each timestep
        """
        if self.samples is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Use the mean of the posterior distributions
        transition_matrix = self.samples['transition_matrix'].mean(axis=0)
        mu = self.samples['mu'].mean(axis=0)
        sigma = self.samples['sigma'].mean(axis=0)

        # Forward algorithm
        forward = jnp.zeros((self.num_timesteps, self.num_states))
        forward = forward.at[0].set(MultivariateNormal(mu, jnp.diag(sigma)).log_prob(data[0]) + 
                                    jnp.log(self.samples['initial_state'].mean(axis=0)))

        for t in range(1, self.num_timesteps):
            forward = forward.at[t].set(MultivariateNormal(mu, jnp.diag(sigma)).log_prob(data[t]) + 
                                        jnp.logsumexp(forward[t-1] + jnp.log(transition_matrix.T), axis=1))

        # Normalize to get probabilities
        state_probs = jnp.exp(forward - jnp.logsumexp(forward, axis=1, keepdims=True))
        return state_probs

    def interpret_states(self, state_probs, threshold=0.6):
        """
        Interpret the cognitive states based on state probabilities.

        Args:
            state_probs (np.ndarray): State probabilities from get_state_probabilities
            threshold (float): Probability threshold for state assignment

        Returns:
            list: Interpreted cognitive states for each timestep
        """
        cognitive_states = ['Attention', 'Relaxation', 'Cognitive Load']
        interpreted_states = []
        for probs in state_probs:
            if np.max(probs) > threshold:
                interpreted_states.append(cognitive_states[np.argmax(probs)])
            else:
                interpreted_states.append('Uncertain')
        return interpreted_states