import numpy as np
import pandas as pd
from scipy import stats
from bayesian_network import BayesianNetwork
from uncertainty_quantifier import UncertaintyQuantifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import time
import os
import networkx as nx

def generate_synthetic_data(n_samples):
    A = np.random.normal(0, 1, n_samples)
    B = A + 0.5 * A**2 + np.random.normal(0, 0.5, n_samples)
    C = np.sin(B) + np.random.normal(0, 0.1, n_samples)
    D = np.exp(0.1 * C) + np.random.normal(0, 0.1, n_samples)
    E = 0.5 * A + 0.5 * C + np.random.normal(0, 0.1, n_samples)
    return pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D, 'E': E})

def parallel_sensitivity_analysis(args):
    bn, node, other_nodes = args
    uq = UncertaintyQuantifier(bn)
    return node, uq.sensitivity_analysis(node, other_nodes)

if __name__ == "__main__":
    start_time = time.time()

    # Generate a larger synthetic dataset
    print("Generating synthetic data...")
    data = generate_synthetic_data(n_samples=100000)

    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Create and train the Bayesian Network
    print("Creating and training Bayesian Network...")
    bn = BayesianNetwork()

    # Define prior knowledge
    prior_edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('A', 'E'), ('C', 'E')]

    # Learn structure and fit parameters
    bn.learn_structure(train_data, prior_edges, method='hill_climb', max_parents=3)
    bn.fit(train_data)

    # Print the learned structure
    print("\nLearned Network Structure:")
    for node, node_obj in bn.nodes.items():
        parents = [parent.name for parent in node_obj.parents]
        children = [child.name for child in node_obj.children]
        print(f"Node {node}:")
        print(f"  Parents: {parents}")
        print(f"  Children: {children}")

    # Perform inference
    uq = UncertaintyQuantifier(bn)

    print("\nEntropy of each node:")
    for node in bn.nodes:
        entropy = uq.entropy(node)
        print(f"  {node}: {entropy:.4f}")

    print("\nMutual Information:")
    for node1 in bn.nodes:
        for node2 in bn.nodes:
            if node1 < node2:
                mi = uq.mutual_information(node1, node2)
                print(f"  MI({node1}, {node2}): {mi:.4f}")

    # Parallel sensitivity analysis
    print("\nPerforming parallel sensitivity analysis...")
    with ProcessPoolExecutor() as executor:
        sensitivity_args = [(bn, node, [n for n in bn.nodes if n != node]) for node in bn.nodes]
        results = executor.map(parallel_sensitivity_analysis, sensitivity_args)

    for node, sensitivities in results:
        print(f"  Sensitivities for {node}:")
        for input_node, sensitivity in sensitivities.items():
            print(f"    {input_node}: {sensitivity:.4f}")

    # Cross-validation
    print("\nPerforming cross-validation...")
    mean_ll, std_ll = bn.cross_validate(train_data)
    print(f"Cross-validation results:")
    print(f"  Mean log-likelihood: {mean_ll:.4f}")
    print(f"  Std log-likelihood: {std_ll:.4f}")

    # Prediction on test set
    print("\nMaking predictions on test set...")
    test_ll = bn.log_likelihood(test_data)
    print(f"Test set log-likelihood: {test_ll:.4f}")

    # Generate samples from the learned model
    print("\nGenerating samples from the learned model...")
    n_samples = 10000
    sampled_data = pd.DataFrame({node: bn.sample_node(node, n_samples) for node in bn.nodes})

    # Create a directory for output files
    os.makedirs("output", exist_ok=True)

    # Visualize the learned structure
    plt.figure(figsize=(10, 8))
    nx.draw(bn.graph, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=12, font_weight='bold')
    plt.title("Learned Bayesian Network Structure")
    plt.savefig("output/learned_structure.png")
    plt.close()

    # Plot pairwise relationships
    sns.pairplot(sampled_data)
    plt.savefig("output/pairwise_relationships.png")
    plt.close()

    # Compare original and sampled distributions
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