import time
import pandas as pd
from sklearn.model_selection import train_test_split
from bayesian_network import BayesianNetwork
import numpy as np


def generate_synthetic_data(n_samples):
    A = np.random.normal(0, 1, n_samples)
    B = A + 0.5 * A**2 + np.random.normal(0, 0.5, n_samples)
    C = np.sin(B) + np.random.normal(0, 0.1, n_samples)
    D = np.exp(0.1 * C) + np.random.normal(0, 0.1, n_samples)
    E = 0.5 * A + 0.5 * C + np.random.normal(0, 0.1, n_samples)
    return pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D, 'E': E})

if __name__ == "__main__":
    start_time = time.time()

    # Generate synthetic data
    print("Generating synthetic data...")
    data = generate_synthetic_data(n_samples=5000)

    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Initialize Bayesian Network
    print("Creating and training Bayesian Network...")
    bn = BayesianNetwork(method='pc', max_parents=3)
    bn.fit(train_data)

    # Visualize the learned structure
    print("Visualizing learned network structure...")
    bn.visualize("output/learned_structure.png")

    # Perform inference on test set
    print("Performing inference on test set...")
    test_ll = bn.log_likelihood(test_data)
    print(f"Test set log-likelihood: {test_ll:.4f}")

    # Cross-validation
    print("Performing cross-validation...")
    mean_ll, std_ll = bn.cross_validate(data)
    print(f"Cross-validation results: Mean Log-Likelihood = {mean_ll:.4f}, Std = {std_ll:.4f}")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
