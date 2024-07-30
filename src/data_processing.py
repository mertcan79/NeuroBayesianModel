import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import List, Tuple, Dict
from scipy import stats
from scipy.stats import norm

def optimize_data_types(df: pd.DataFrame) -> pd.DataFrame:
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

def normalize_skewed_features(data, columns, threshold=1):
    for col in columns:
        if abs(stats.skew(data[col])) > threshold:
            data[col] = stats.boxcox(data[col] - data[col].min() + 1)[0]
    return data

def simple_outlier_treatment(data: pd.DataFrame, columns: List[str], threshold: float = 1.5) -> pd.DataFrame:
    for col in columns:
        if col in data.columns:
            col_data = data[col].dropna()

            if len(col_data) == 0:
                print(f"Warning: Column {col} contains only non-finite values. Skipping.")
                continue

            # Calculate IQR
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1

            # Determine outliers
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            mask = (data[col] < lower_bound) | (data[col] > upper_bound)
            finite_mask = mask & np.isfinite(data[col])
            n_finite_outliers = finite_mask.sum()

            if n_finite_outliers > 0:
                replacement_value = col_data.mean()
                # Cast replacement_value to the same dtype as the column
                replacement_value = replacement_value.astype(data[col].dtype)
                data.loc[finite_mask, col] = replacement_value

            if mask.sum() != n_finite_outliers:
                print(f"Warning: {mask.sum() - n_finite_outliers} non-finite outliers in column {col} were not replaced.")

    return data

def detect_and_handle_outliers(data: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> pd.DataFrame:
    for column in columns:
        if column in data.columns:
            mean = data[column].mean()
            std = data[column].std()
            outliers = abs(data[column] - mean) > (threshold * std)
            data.loc[outliers, column] = np.clip(data.loc[outliers, column], mean - threshold * std, mean + threshold * std)
    return data


def add_interaction_terms(data: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            data[f"{features[i]}_{features[j]}_interaction"] = data[features[i]] * data[features[j]]
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

    data = detect_and_handle_outliers(data, numeric_columns)
    #data = simple_outlier_treatment(data, numeric_columns)
    
    # Handle numeric columns
    numeric_imputer = SimpleImputer(strategy='median')
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].mean())
    data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])
    
    # Handle categorical columns
    for col in categorical_columns:
        if col in data.columns:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            data[[col]] = categorical_imputer.fit_transform(data[[col]])

    #data = transform_skewed_features(data)
    data = normalize_skewed_features(data, numeric_columns)
    # Apply StandardScaler to numeric columns
    scaler = RobustScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    

    return data

def prepare_data(behavioral_path: str, hcp_path: str,
                  behavioral_features: List[str], hcp_features: List[str],
                  categorical_columns: List[str], index: str, interaction_features: List[str]) -> Tuple[pd.DataFrame, List[str], Dict[str, List]]:
    behavioral, hcp = load_data(behavioral_path, hcp_path)
    
    behavioral = optimize_data_types(behavioral)
    hcp = optimize_data_types(hcp)
    
    behavioral, hcp = select_features(behavioral, hcp, behavioral_features, hcp_features)
    
    data = pd.merge(hcp, behavioral, on="Subject")
    
    processed_data = preprocess_data(data, categorical_columns, index)
    
    categories = {}
    for col in categorical_columns:
        if col in processed_data.columns:
            categories[col] = sorted(processed_data[col].dropna().unique().tolist())
    
    processed_data = add_interaction_terms(processed_data, interaction_features)

    return processed_data, categorical_columns
