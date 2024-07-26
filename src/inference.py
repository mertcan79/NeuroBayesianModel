import numpy as np
import pandas as pd
from typing import Dict, Any, List
from bayesian_node import BayesianNode, CategoricalNode
from scipy.stats import multinomial, norm, chi2_contingency

class Inference:
    def __init__(self, nodes):
        self.nodes = nodes
        print(f"Initialized Inference with {len(nodes)} nodes")

    def gibbs_sampling(self, num_samples: int, burn_in: int = 1000):
        samples = {node: np.zeros(num_samples + burn_in) for node in self.nodes}
        
        # Initialize with random values
        for node_name, node in self.nodes.items():
            if isinstance(node, CategoricalNode):
                samples[node_name][0] = np.random.choice(node.categories)
            else:
                samples[node_name][0] = np.random.normal()

        # Gibbs sampling
        for i in range(1, num_samples + burn_in):
            for node_name, node in self.nodes.items():
                parent_values = [samples[parent.name][i-1] for parent in node.parents]
                if isinstance(node, CategoricalNode):
                    probs = node.get_conditional_probs(tuple(parent_values))
                    samples[node_name][i] = np.random.choice(node.categories, p=probs)
                else:
                    mean = node.get_conditional_mean(parent_values)
                    std = node.distribution['std']
                    samples[node_name][i] = np.random.normal(mean, std)

        # Discard burn-in samples
        return {node: samples[node][burn_in:] for node in samples}

    def compute_sensitivity(self, target_node: str, num_samples: int = 1000) -> Dict[str, float]:
        print(f"Computing sensitivity for {target_node}")
        samples = self.gibbs_sampling(num_samples)
        target_samples = samples[target_node]
        
        sensitivities = {}
        for node, node_samples in samples.items():
            if node != target_node:
                if isinstance(self.nodes[node], CategoricalNode) or isinstance(self.nodes[target_node], CategoricalNode):
                    sensitivities[node] = self._compute_categorical_correlation(target_samples, node_samples)
                else:
                    sensitivities[node] = np.corrcoef(target_samples, node_samples)[0, 1]
        
        print(f"Computed sensitivities for {target_node}")
        return sensitivities

    def _compute_categorical_correlation(self, x, y):
        contingency = pd.crosstab(x, y)
        chi2 = chi2_contingency(contingency)[0]
        n = len(x)
        phi2 = chi2 / n
        r, k = contingency.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        
        denominator = min((kcorr-1), (rcorr-1))
        if denominator > 0:
            return np.sqrt(phi2corr / denominator)
        else:
            return 0.0