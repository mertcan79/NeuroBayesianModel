import numpy as np
import pandas as pd
from bayesian_network import BayesianNetwork
from sklearn.model_selection import train_test_split
import time

def generate_synthetic_data(n_samples=5000):
    A = np.random.normal(0, 1, n_samples)
    B = 0.5 * A + np.random.normal(0, 0.5, n_samples)
    C = 0.3 * A + 0.7 * B + np.random.normal(0, 0.3, n_samples)
    return pd.DataFrame({'A': A, 'B': B, 'C': C})

data = generate_synthetic_data()

def progress_callback(progress):
    print(f"Progress: {progress*100:.0f}%")

def print_section(title):
    print(f"\n{'-'*10} {title} {'-'*10}")

if __name__ == "__main__":
    start_time = time.time()

    print_section("Data Generation")
    data = generate_synthetic_data(n_samples=5000)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Data profile: {len(data)} samples, {len(data.columns)} features")

    print_section("Bayesian Network Training")
    bn = BayesianNetwork(method='hill_climb', max_parents=2)
    prior_edges = [('A', 'B'), ('A', 'C'), ('B', 'C')]
    bn.fit(train_data, prior_edges, progress_callback=progress_callback)

    print_section("Model Overview")
    num_nodes = len(bn.nodes)
    print(f"Number of nodes: {num_nodes}")
    for node_name, node in bn.nodes.items():
        print(f"Node {node_name}: Parents = {[p.name for p in node.parents]}")

    print_section("Evaluation Metrics")
    test_ll = bn.log_likelihood(test_data)
    mean_ll, std_ll = bn.cross_validate(data, k_folds=5)
    print(f"Test log-likelihood: {test_ll:.4f}")
    print(f"Cross-validation: Mean LL = {mean_ll:.4f}, Std = {std_ll:.4f}")

    print_section("Sensitivity Analysis (Top 3)")
    sensitivity = bn.compute_sensitivity('C', num_samples=5000)
    top_sensitivities = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)[:3]
    for node, value in top_sensitivities:
        print(f"Sensitivity of C to {node}: {value:.4f}")

    print_section("Metropolis-Hastings Sampling (Sample Mean & Std)")
    observed_data = {'A': 0.5}
    mh_samples = bn.metropolis_hastings(observed_data, num_samples=5000)
    for node, samples in mh_samples.items():
        if node not in observed_data:
            print(f"{node}: Mean = {np.mean(samples):.4f}, Std = {np.std(samples):.4f}")

    print_section("Summary")
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
