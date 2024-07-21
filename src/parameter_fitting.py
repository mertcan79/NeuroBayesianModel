import numpy as np
import pandas as pd
from typing import Dict
from scipy import stats
import logging
from sklearn.impute import SimpleImputer
from bayesian_node import BayesianNode, CategoricalNode

logger = logging.getLogger(__name__)

def fit_parameters(nodes: Dict[str, BayesianNode], data: pd.DataFrame):
    for node_name, node in nodes.items():
        node_data = data[node_name].values
        node.fit_scaler(node_data)
        scaled_node_data = node.transform(node_data)
        
        if not node.parents:
            if isinstance(node, CategoricalNode):
                unique, counts = np.unique(node_data, return_counts=True)
                probabilities = counts / len(node_data)
                node.set_distribution(probabilities)
            else:
                mean = np.mean(scaled_node_data)
                std = np.std(scaled_node_data)
                node.set_distribution(stats.norm, params={'loc': mean, 'scale': std})
        else:
            parent_data = np.column_stack([nodes[p.name].transform(data[p.name].values) for p in node.parents])
            
            # Ensure the parent data is numeric
            if parent_data.dtype.kind in 'O':
                raise ValueError(f"Parent data for node {node_name} contains non-numeric data.")
            
            # Convert categorical parent data to numeric form if needed
            for i, parent in enumerate(node.parents):
                if isinstance(nodes[parent.name], CategoricalNode):
                    parent_data[:, i] = nodes[parent.name].transform(parent_data[:, i])
            
            beta = np.linalg.lstsq(parent_data, scaled_node_data, rcond=None)[0]
            residuals = scaled_node_data - parent_data.dot(beta)
            std = np.std(residuals)
            node.set_distribution(stats.norm, params={'loc': 0, 'scale': std, 'beta': beta})

            # Save additional parameters for linear regression
            node.params['intercept'] = np.mean(scaled_node_data - parent_data.dot(beta))



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

