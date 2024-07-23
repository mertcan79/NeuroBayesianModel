import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from typing import List, Callable, Tuple, Dict, Any

from .bayesian_network import BayesianNetwork

logger = logging.getLogger(__name__)

class BayesianModel:
    def __init__(self, method='hill_climb', max_parents=4, iterations=1000, categorical_columns=None):
        self.network = BayesianNetwork(method=method, max_parents=max_parents, iterations=iterations, categorical_columns=categorical_columns)


    def write_results_to_json(self, results):
        self.network.write_results_to_json(results)

    def explain_structure_extended(self):
        return self.network.explain_structure_extended()

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for the Bayesian network."""
        return self.network.preprocess_data(data)

    def fit(self, data: pd.DataFrame, prior_edges: List[tuple] = None, progress_callback: Callable[[float], None] = None):
        """Fit the Bayesian network to the data."""
        try:
            preprocessed_data = self.preprocess_data(data)
            self.network.fit(preprocessed_data, prior_edges=prior_edges, progress_callback=progress_callback)
        except Exception as e:
            logger.error(f"Error during model fitting: {e}")
            raise

    def evaluate(self, data: pd.DataFrame, k_folds: int = 5) -> Tuple[float, float]:
        """Evaluate the Bayesian network using cross-validation."""
        preprocessed_data = self.preprocess_data(data)
        return self.network.cross_validate(preprocessed_data, k_folds=k_folds)

    def simulate_intervention(self, interventions: Dict[str, Any], size: int = 1000) -> pd.DataFrame:
        """Simulate interventions on the Bayesian network."""
        return self.network.simulate_intervention(interventions, size=size)

    def save(self, filename: str):
        """Save the Bayesian network to a file."""
        self.network.save(filename)

    @classmethod
    def load(cls, filename: str):
        """Load the Bayesian network from a file."""
        network = BayesianNetwork.load(filename)
        model = cls(method=network.method, max_parents=network.max_parents, iterations=network.iterations, categorical_columns=network.categorical_columns)
        model.network = network
        return model

    def compute_sensitivity(self, target_node: str, num_samples: int = 1000) -> Dict[str, float]:
        """Compute sensitivity of the target node to changes in other nodes."""
        return self.network.compute_sensitivity(target_node, num_samples=num_samples)
