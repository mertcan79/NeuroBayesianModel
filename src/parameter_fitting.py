import numpy as np
import pandas as pd
from typing import Dict
from scipy import stats
from bayesian_node import BayesianNode
import logging
from sklearn.impute import SimpleImputer

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


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    categorical_columns = data.select_dtypes(exclude=[np.number]).columns
    
    numeric_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    if len(numeric_columns) > 0:
        data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])
    
    if len(categorical_columns) > 0:
        data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])
    
    return data

