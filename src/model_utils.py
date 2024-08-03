import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def simple_linear_model(X, y=None):
    beta = numpyro.sample("beta", dist.Normal(0, 1).expand([X.shape[1]]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    mean = jnp.dot(X, beta)
    numpyro.sample("y", dist.Normal(mean, sigma), obs=y)

def nonlinear_cognitive_model(X, y=None):
    num_samples, num_features = X.shape

    # Reduce number of hidden units
    W1 = numpyro.sample("W1", dist.Normal(0, 1).expand([num_features, 5]))  # Changed from 10 to 5
    b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([5]))
    W2 = numpyro.sample("W2", dist.Normal(0, 1).expand([5, 1]))
    b2 = numpyro.sample("b2", dist.Normal(0, 1))

    hidden = jnp.maximum(0, jnp.dot(X, W1) + b1)
    y_hat = jnp.dot(hidden, W2) + b2

    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    numpyro.sample("y", dist.Normal(y_hat, sigma), obs=y)
