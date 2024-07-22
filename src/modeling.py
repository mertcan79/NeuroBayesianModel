from bayesian_network import BayesianNetwork
import pandas as pd
from typing import List, Tuple, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_bayesian_network(data: pd.DataFrame, categorical_columns: List[str], prior_edges: List[Tuple[str, str]]) -> BayesianNetwork:
    model = BayesianNetwork(method='hill_climb', max_parents=4, categorical_columns=categorical_columns)
    model.fit(data, prior_edges=prior_edges)
    return model

def analyze_network(model: BayesianNetwork, data: pd.DataFrame) -> Dict[str, Any]:
    results = {}

    logger.info("Computing log-likelihood")
    results['log_likelihood'] = model.log_likelihood(data)

    logger.info("Performing cross-validation")
    mean_ll, std_ll = model.cross_validate(data, k_folds=4)
    results['cross_validation'] = {'mean': mean_ll, 'std': std_ll}

    logger.info("Computing sensitivity")
    sensitivity = model.compute_sensitivity('CogFluidComp_Unadj', num_samples=600)
    results['sensitivity'] = dict(sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)[:10])

    logger.info("Explaining network structure")
    results['network_structure'] = model.explain_structure_extended()

    logger.info("Performing Metropolis-Hastings sampling")
    observed_data = {'Age': 0, 'Gender': 1, 'MMSE_Score': 0}
    mh_samples = model.metropolis_hastings(observed_data, num_samples=1000)
    results['mh_samples'] = {node: {"mean": float(np.mean(samples)), "std": float(np.std(samples))} 
                             for node, samples in mh_samples.items() if node not in observed_data}

    return results

def create_hierarchical_bayesian_network(data: pd.DataFrame, categorical_columns: List[str], hierarchical_levels: List[str], level_constraints: Dict[str, List[str]]) -> BayesianNetwork:
    from bayesian_network import HierarchicalBayesianNetwork
    h_model = HierarchicalBayesianNetwork(levels=hierarchical_levels, method='hill_climb', max_parents=3, categorical_columns=categorical_columns)
    h_model.fit(data, level_constraints=level_constraints)
    return h_model