import jax.numpy as jnp
from jax import random, vmap
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from base import BaseBayesianNetwork
import jax

class DynamicBayesianNetwork(BaseBayesianNetwork):
    def __init__(self, num_features, num_states=2, iterations=100):
        super().__init__()
        self.num_features = num_features
        self.num_states = num_states
        self.iterations = iterations
        self.samples = None

    def model(self, data):
        num_timesteps, _ = data.shape

        # Initial state probabilities
        initial_probs = numpyro.sample('initial_probs', dist.Dirichlet(jnp.ones(self.num_states)))

        # Transition matrix
        trans_matrix = numpyro.sample('trans_matrix', 
                                      dist.Dirichlet(jnp.ones(self.num_states)).expand([self.num_states]))

        # Emission means and scales
        means = numpyro.sample('means', dist.Normal(0, 1).expand([self.num_states, self.num_features]))
        scales = numpyro.sample('scales', dist.HalfNormal(1).expand([self.num_states, self.num_features]))

        def emission(state, t):
            return numpyro.sample(f'obs_{t}', dist.Normal(means[state], scales[state]), obs=data[t])

        state = numpyro.sample('state_0', dist.Categorical(initial_probs))
        emission(state, 0)

        for t in range(1, num_timesteps):
            state = numpyro.sample(f'state_{t}', dist.Categorical(trans_matrix[state]))
            emission(state, t)

    def fit(self, data):
        rng_key = random.PRNGKey(0)
        kernel = NUTS(self.model)
        mcmc = MCMC(kernel, num_warmup=50, num_samples=100)
        mcmc.run(rng_key, data=data)
        self.samples = mcmc.get_samples()
        return self.samples

    def predict(self, data):
        if self.samples is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        trans_matrix = self.samples['trans_matrix'].mean(axis=0)
        means = self.samples['means'].mean(axis=0)
        scales = self.samples['scales'].mean(axis=0)
        initial_probs = self.samples['initial_probs'].mean(axis=0)

        num_timesteps = len(data)
        log_probs = vmap(lambda obs: vmap(lambda mu, scale: dist.Normal(mu, scale).log_prob(obs).sum())(means, scales))(data)
        
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

        for t in range(num_timesteps - 2, -1, -1):
            best_path = best_path.at[t].set(paths[t, best_path[t+1]])

        return best_path
        
        jax.lax.scan(backtrace, best_path[-1], jnp.arange(num_timesteps - 2, -1, -1))
        return best_path

    def get_state_probabilities(self, data):
        if self.samples is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        trans_matrix = self.samples['trans_matrix'].mean(axis=0)
        means = self.samples['means'].mean(axis=0)
        scales = self.samples['scales'].mean(axis=0)
        initial_probs = self.samples['initial_probs'].mean(axis=0)

        num_timesteps = len(data)
        log_probs = vmap(lambda obs: vmap(lambda mu, scale: dist.Normal(mu, scale).log_prob(obs).sum())(means, scales))(data)

        def forward_step(carry, t):
            prev_logp = carry
            curr_logp = logsumexp(prev_logp[:, None] + jnp.log(trans_matrix), axis=0) + log_probs[t]
            return curr_logp, curr_logp

        init_logp = jnp.log(initial_probs) + log_probs[0]
        _, forward_probs = jax.lax.scan(forward_step, init_logp, jnp.arange(1, num_timesteps))
        forward_probs = jnp.vstack([init_logp[None, :], forward_probs])

        return jnp.exp(forward_probs - logsumexp(forward_probs, axis=1, keepdims=True))

    def interpret_states(self, state_probs, threshold=0.6):
        cognitive_states = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
        interpreted_states = []
        for probs in state_probs:
            if np.max(probs) > threshold:
                interpreted_states.append(cognitive_states[np.argmax(probs)])
            else:
                interpreted_states.append('Uncertain')
        return interpreted_states

    def compute_edge_probabilities(self):
        if self.samples is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        edge_probs = {}

        trans_matrix = self.samples['trans_matrix'].mean(axis=0)
        for i in range(self.num_states):
            for j in range(self.num_states):
                edge_probs[f'State_{i}->State_{j}'] = trans_matrix[i, j]

        means = self.samples['means'].mean(axis=0)
        for i in range(self.num_states):
            for j in range(self.num_features):
                edge_probs[f'State_{i}->Feature_{j}'] = abs(means[i, j])

        return edge_probs