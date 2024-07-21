import numpy as np
import pandas as pd
from typing import Dict
from scipy import stats
import logging
from sklearn.impute import SimpleImputer
from bayesian_node import BayesianNode, CategoricalNode

logger = logging.getLogger(__name__)

def fit_parameters(nodes, data):
    for node_name, node in nodes.items():
        node_data = pd.Series(data[node_name]).apply(pd.to_numeric, errors='coerce').dropna()

        if isinstance(node, CategoricalNode):
            node_data = node.transform(node_data)
            counts = np.bincount(node_data)
            node.set_distribution(counts)
        else:
            # Fit the scaler and distribution
            node.fit_scaler(node_data.values)
            scaled_node_data = node_data.values
            
            if node.parents:
                parent_data_list = []
                for parent in node.parents:
                    parent_data = pd.Series(data[parent.name]).apply(pd.to_numeric, errors='coerce').dropna()
                    parent_data_list.append(parent_data)
                
                parent_data_df = pd.DataFrame(parent_data_list).T
                parent_data_df, scaled_node_data = parent_data_df.align(pd.Series(scaled_node_data), join='inner', axis=0)
                
                parent_data = parent_data_df.values
                scaled_node_data = scaled_node_data.values
                
                if parent_data.shape[0] != scaled_node_data.shape[0]:
                    raise ValueError("Mismatch in number of rows between parent data and node data.")
                
                parent_data = parent_data.astype(np.float64)
                scaled_node_data = scaled_node_data.astype(np.float64)
                
                beta = np.linalg.lstsq(parent_data, scaled_node_data, rcond=None)[0]
                residuals = scaled_node_data - parent_data.dot(beta)
                std = np.std(residuals)
                
                # Set a normal distribution with parameters
                node.set_distribution(stats.norm, params={'loc': np.mean(scaled_node_data), 'scale': std})


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

