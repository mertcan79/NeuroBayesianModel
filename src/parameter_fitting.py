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
        node_data = data[node_name]
        parent_data = None
        if node.parents:
            parent_data = data[[parent.name for parent in node.parents]]
        
        if isinstance(node, CategoricalNode):
            scaled_node_data = node.transform(node_data.values)
        else:
            scaled_node_data = node_data.values

        node.fit(scaled_node_data, parent_data)

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

