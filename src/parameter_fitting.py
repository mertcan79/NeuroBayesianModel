import numpy as np
import pandas as pd
from typing import Dict
from scipy import stats
from bayesian_node import BayesianNode
import logging

logger = logging.getLogger(__name__)

def fit_parameters(nodes: Dict[str, BayesianNode], data: pd.DataFrame):
    for node_name, node in nodes.items():
        node_data = data[node_name].values
        node.fit_scaler(node_data)
        scaled_node_data = node.transform(node_data)

        if not node.parents:
            mean, std = np.mean(scaled_node_data), np.std(scaled_node_data)
            node.set_distribution(stats.norm, params={'loc': mean, 'scale': std})
        else:
            parent_data = np.column_stack([nodes[p.name].transform(data[p.name].values) for p in node.parents])
            beta = np.linalg.lstsq(parent_data, scaled_node_data, rcond=None)[0]
            residuals = scaled_node_data - parent_data.dot(beta)
            std = np.std(residuals)
            node.set_distribution(stats.norm, params={'beta': beta, 'scale': std})
        
        logger.info(f"Fitted parameters for node {node_name}: {node.params}")
