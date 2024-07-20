import numpy as np
import pandas as pd
from bayesian_network import BayesianNetwork
from sklearn.model_selection import train_test_split
import time

def generate_synthetic_data(n_samples):
    A = np.random.normal(0, 1, n_samples)
    B = 0.5 * A + np.random.normal(0, 0.5, n_samples)
    C = 0.3 * A + 0.5 * B + np.random.normal(0, 0.5, n_samples)
    return pd.DataFrame({'A': A, 'B': B, 'C': C})

def progress_callback(progress):
    print(f"Progress: {progress*100:.0f}%")

if __name__ == "__main__":
    start_time = time.time()

    # Generate synthetic data
    print("Generating synthetic data...")
    data = generate_synthetic_data(n_samples=1000)  # Reduced to 1000 samples

    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Create and train the Bayesian Network
    print("Creating and training Bayesian Network...")
    bn = BayesianNetwork(method='hill_climb', max_parents=2)

    # Define prior knowledge
    prior_edges = [('A', 'B'), ('A', 'C'), ('B', 'C')]

    # Fit the network
    bn.fit(train_data, prior_edges, progress_callback=progress_callback)

    # Print learned structure
    print("\nLearned Network Structure:")
    for node, node_data in bn.nodes.items():
        parents = node_data.get('parents', [])
        children = node_data.get('children', [])
        print(f"Node {node}:")
        print(f"  Parents: {parents}")
        print(f"  Children: {children}")

    # Perform inference on test set
    print("\nPerforming inference on test set...")
    test_ll = bn.log_likelihood(test_data)
    print(f"Test set log-likelihood: {test_ll:.4f}")

    # Cross-validation
    print("\nPerforming cross-validation...")
    mean_ll, std_ll = bn.cross_validate(data, k_folds=5)
    print(f"Cross-validation results:")
    print(f"  Mean log-likelihood: {mean_ll:.4f}")
    print(f"  Std log-likelihood: {std_ll:.4f}")

    # Generate samples from the learned model
    print("\nGenerating samples from the learned model...")
    n_samples = 1000
    sampled_data = pd.DataFrame({node: bn.sample_node(node, n_samples) for node in bn.nodes})

    # Calculate mean and standard deviation of sampled data
    print("\nSampled data statistics:")
    for column in sampled_data.columns:
        mean = sampled_data[column].mean()
        std = sampled_data[column].std()
        print(f"  {column}: Mean = {mean:.4f}, Std = {std:.4f}")

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")