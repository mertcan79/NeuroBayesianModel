import numpy as np
import pandas as pd
from bayesian_network import BayesianNetwork
from sklearn.model_selection import train_test_split
import time

def generate_synthetic_data(n_samples=25000, n_vars=15):
    data = pd.DataFrame()
    data['A'] = np.random.normal(0, 1, n_samples)
    data['B'] = 0.5 * data['A'] + np.random.normal(0, 0.5, n_samples)
    data['C'] = 0.3 * data['A'] + 0.5 * data['B'] + np.random.normal(0, 0.5, n_samples)
    for i in range(3, n_vars):
        parents = np.random.choice(range(i), size=min(3, i), replace=False)
        data[f'Var_{i}'] = sum(0.3 * data.iloc[:, p] for p in parents) + np.random.normal(0, 0.5, n_samples)
    return data

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
    for node_name, node in bn.nodes.items():
        print(f"Node {node_name}:")
        print(f"  Parents: {[parent.name for parent in node.parents]}")
        print(f"  Children: {[child.name for child in node.children]}")

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