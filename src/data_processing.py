import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict
from scipy import stats

def optimize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert int64 to int32 and float64 to float32 to save memory.
    """
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            if df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')
        elif pd.api.types.is_float_dtype(df[col]):
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
    return df

def load_data(behavioral_path: str, hcp_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    behavioral = pd.read_csv(behavioral_path)
    hcp = pd.read_csv(hcp_path)
    return behavioral, hcp

def select_features(behavioral: pd.DataFrame, hcp: pd.DataFrame,
                     behavioral_features: List[str], hcp_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hcp = hcp[hcp_features].copy()
    behavioral = behavioral[behavioral_features].copy()
    return behavioral, hcp

def transform_skewed_features(data: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    for col in data.select_dtypes(include=[np.number]).columns:
        skewness = stats.skew(data[col].dropna())
        if abs(skewness) > threshold:
            data[col] = np.log1p(data[col] - data[col].min())
    return data

def preprocess_data(data: pd.DataFrame, categorical_columns: List[str], index: str = None) -> pd.DataFrame:
    data = data.copy()
    
    # Set 'Subject' as index
    if index:
        data.set_index(index, inplace=True)
    
    # Encode categorical columns
    for col in categorical_columns:
        if col in data.columns:
            data[col] = pd.Categorical(data[col]).codes
    
    # Identify numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in categorical_columns]
    
    # Handle numeric columns
    numeric_imputer = SimpleImputer(strategy='median')
    data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])
    
    # Handle categorical columns
    for col in categorical_columns:
        if col in data.columns:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            data[[col]] = categorical_imputer.fit_transform(data[[col]])

    data = transform_skewed_features(data)
    # Apply StandardScaler to numeric columns
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    return data

def prepare_data(behavioral_path: str, hcp_path: str,
                  behavioral_features: List[str], hcp_features: List[str],
                  categorical_columns: List[str], index: str) -> Tuple[pd.DataFrame, List[str], Dict[str, List]]:
    behavioral, hcp = load_data(behavioral_path, hcp_path)
    
    behavioral = optimize_data_types(behavioral)
    hcp = optimize_data_types(hcp)

    behavioral, hcp = select_features(behavioral, hcp, behavioral_features, hcp_features)
    
    data = pd.merge(hcp, behavioral, on="Subject")
    
    processed_data = preprocess_data(data, categorical_columns, index)
    
    # Get categories for categorical variables
    categories = {}
    for col in categorical_columns:
        if col in processed_data.columns:
            categories[col] = sorted(processed_data[col].dropna().unique().tolist())
    
    return processed_data, categorical_columns, categories