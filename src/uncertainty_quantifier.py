import numpy as np
from typing import List, Dict
from bayesian_network import BayesianNetwork
from scipy.stats import entropy as scipy_entropy
from scipy.stats import gaussian_kde

class UncertaintyQuantifier:
    def __init__(self, network: BayesianNetwork):
        self.network = network

    def entropy(self, node_name: str, n_samples: int = 10000) -> float:
        samples = self.network.sample_node(node_name, n_samples)
        kde = gaussian_kde(samples)
        return scipy_entropy(kde(samples))

    def mutual_information(self, node1_name: str, node2_name: str, n_samples: int = 10000) -> float:
        samples1 = self.network.sample_node(node1_name, n_samples)
        samples2 = self.network.sample_node(node2_name, n_samples)
        
        kde1 = gaussian_kde(samples1)
        kde2 = gaussian_kde(samples2)
        kde_joint = gaussian_kde(np.vstack([samples1, samples2]))
        
        mutual_info = np.mean(np.log(kde_joint(np.vstack([samples1, samples2])) / (kde1(samples1) * kde2(samples2))))
        return mutual_info

    def sensitivity_analysis(self, target_node: str, input_nodes: List[str], n_samples: int = 1000) -> Dict[str, float]:
        sensitivities = {}
        base_output = self.network.sample_node(target_node, size=n_samples)
        
        for input_node in input_nodes:
            perturbed_network = self.network.copy()
            perturbed_data = perturbed_network.sample_node(input_node, size=n_samples) * 1.1
            perturbed_network.nodes[input_node].distribution = gaussian_kde(perturbed_data)
            perturbed_output = perturbed_network.sample_node(target_node, size=n_samples)
            sensitivities[input_node] = np.mean(np.abs(perturbed_output - base_output) / np.maximum(np.abs(base_output), 1e-10))
        
        return sensitivities

    def conditional_entropy(self, node: str, given_nodes: List[str], n_samples: int = 10000) -> float:
        samples = {node: self.network.sample_node(node, n_samples) for node in [node] + given_nodes}
        kde = gaussian_kde(np.vstack(list(samples.values())))
        return -np.mean(np.log(kde(np.vstack(list(samples.values())))))

    def kullback_leibler_divergence(self, node1: str, node2: str, n_samples: int = 10000) -> float:
        samples1 = self.network.sample_node(node1, n_samples)
        samples2 = self.network.sample_node(node2, n_samples)
        kde1 = gaussian_kde(samples1)
        kde2 = gaussian_kde(samples2)
        return np.mean(np.log(kde1(samples1) / kde2(samples1)))