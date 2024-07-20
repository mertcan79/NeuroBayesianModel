import numpy as np
import pandas as pd
from scipy import stats
from bayesian_network import BayesianNetwork
from uncertainty_quantifier import UncertaintyQuantifier
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Generate synthetic data
np.random.seed(42)
n_samples = 5000

A = np.random.normal(0, 1, n_samples)
B = A + 0.5 * A**2 + np.random.normal(0, 0.5, n_samples)
C = np.sin(B) + np.random.normal(0, 0.1, n_samples)
D = np.exp(0.1 * C) + np.random.normal(0, 0.1, n_samples)
E = 0.5 * A + 0.5 * C + np.random.normal(0, 0.1, n_samples)

data = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D, 'E': E})

# Create and train the Bayesian Network
bn = BayesianNetwork()

# Define prior knowledge
prior_edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('A', 'E'), ('C', 'E')]

# Learn structure and fit parameters
bn.learn_structure(data, prior_edges)
bn.fit(data)

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

print("\nSensitivity Analysis:")
for node in bn.nodes:
    sensitivities = uq.sensitivity_analysis(node, [n for n in bn.nodes if n != node])
    print(f"  Sensitivities for {node}:")
    for input_node, sensitivity in sensitivities.items():
        print(f"    {input_node}: {sensitivity:.4f}")

# Cross-validation
mean_ll, std_ll = bn.cross_validate(data)
print(f"\nCross-validation results:")
print(f"  Mean log-likelihood: {mean_ll:.4f}")
print(f"  Std log-likelihood: {std_ll:.4f}")

# Visualize the learned structure
plt.figure(figsize=(10, 8))
nx.draw(bn.graph, with_labels=True, node_color='lightblue', 
        node_size=3000, font_size=12, font_weight='bold')
plt.title("Learned Bayesian Network Structure")
plt.savefig("learned_structure.png")
plt.close()

# Plot pairwise relationships
sns.pairplot(data)
plt.savefig("pairwise_relationships.png")
plt.close()

# Generate samples from the learned model
n_samples = 1000
sampled_data = pd.DataFrame({node: bn.sample_node(node, n_samples) for node in bn.nodes})

# Compare original and sampled distributions
for node in bn.nodes:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data[node], label='Original')
    sns.kdeplot(sampled_data[node], label='Sampled')
    plt.title(f"Distribution Comparison for Node {node}")
    plt.legend()
    plt.savefig(f"distribution_comparison_{node}.png")
    plt.close()