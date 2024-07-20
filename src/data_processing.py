import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # Create a copy of the DataFrame to avoid modifying the original data
    data = data.copy()
    
    # Identify numeric and categorical columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    categorical_columns = data.select_dtypes(exclude=[np.number]).columns
    
    # Initialize imputers
    numeric_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    # Handle numeric columns
    if len(numeric_columns) > 0:
        data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])
    
    # Handle categorical columns
    if len(categorical_columns) > 0:
        data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])
    
    return data
