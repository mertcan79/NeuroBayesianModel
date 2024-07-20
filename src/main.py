import numpy as np
import pandas as pd
from bayesian_network import BayesianNetwork
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

def generate_synthetic_data(n_samples):
    A = np.random.normal(0, 1, n_samples)
    B = A + 0.5 * A**2 + np.random.normal(0, 0.5, n_samples)
    C = np.sin(B) + np.random.normal(0, 0.1, n_samples)
    D = np.exp(0.1 * C) + np.random.normal(0, 0.1, n_samples)
    E = 0.5 * A + 0.5 * C + np.random.normal(0, 0.1, n_samples)
    return pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D, 'E': E})

def progress_callback(progress):
    print(f"Progress: {progress*100:.0f}%")

if __name__ == "__main__":
    start_time = time.time()

    # Generate synthetic data
    print("Generating synthetic data...")
    data = generate_synthetic_data(n_samples=5000)  # Reduced to 5000 samples

    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Create and train the Bayesian Network
    print("Creating and training Bayesian Network...")
    bn = BayesianNetwork(method='hill_climb', max_parents=3)

    # Define prior knowledge
    prior_edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('A', 'E'), ('C', 'E')]

    # Fit the network
    bn.fit(train_data, prior_edges, progress_callback=progress_callback)

    # Visualize the learned structure
    print("\nVisualizing learned network structure...")
    os.makedirs("output", exist_ok=True)
    bn.visualize("output/learned_structure.png")

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
    n_samples = 5000
    sampled_data = pd.DataFrame({node: bn.sample_node(node, n_samples) for node in bn.nodes})

    # Plot pairwise relationships
    print("\nPlotting pairwise relationships...")
    sns.pairplot(sampled_data, plot_kws={'alpha': 0.1})
    plt.savefig("output/pairwise_relationships.png")
    plt.close()

    # Compare original and sampled distributions
    print("\nComparing original and sampled distributions...")
    for node in bn.nodes:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data[node], label='Original')
        sns.kdeplot(sampled_data[node], label='Sampled')
        plt.title(f"Distribution Comparison for Node {node}")
        plt.legend()
        plt.savefig(f"output/distribution_comparison_{node}.png")
        plt.close()

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")