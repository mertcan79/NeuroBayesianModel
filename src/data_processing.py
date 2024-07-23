import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    behavioral = pd.read_csv('/Users/macbookair/Documents/NeuroBayesianModel/data/connectome_behavioral.csv')
    hcp = pd.read_csv('/Users/macbookair/Documents/NeuroBayesianModel/data/hcp_freesurfer.csv')
    return behavioral, hcp

def select_features(behavioral: pd.DataFrame, hcp: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    relevant_features_behavioral = [
        'Subject', 'Age', 'Gender', 'CogFluidComp_Unadj', 'CogCrystalComp_Unadj', 'MMSE_Score',
        'NEOFAC_O', 'NEOFAC_C', 'ProcSpeed_Unadj', 'CardSort_Unadj', 'PicVocab_Unadj', 'ReadEng_Unadj'
    ]

    relevant_features_hcp = [
        'Subject', 'FS_TotCort_GM_Vol', 'FS_SubCort_GM_Vol', 'FS_Total_GM_Vol', 'FS_Tot_WM_Vol', 'FS_BrainStem_Vol',
        'FS_L_Hippo_Vol', 'FS_R_Hippo_Vol', 'FS_L_Amygdala_Vol', 'FS_R_Amygdala_Vol',
        'FS_L_Caudate_Vol', 'FS_R_Caudate_Vol', 'FS_L_Putamen_Vol', 'FS_R_Putamen_Vol',
    ]

    hcp = hcp[relevant_features_hcp].copy()
    behavioral = behavioral[relevant_features_behavioral].copy()

    return behavioral, hcp

def preprocess_data(data: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    data = data.copy()
    
    print("Original data types:")
    print(data.dtypes)
    print("\nMissing values:")
    print(data.isnull().sum())
    
    # Handle 'Age' column
    if 'Age' in data.columns:
        # Convert age ranges to categorical values
        data['Age'] = pd.Categorical(data['Age']).codes
    
    # Handle 'Gender' column
    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].map({'M': 0, 'F': 1})
    
    # Identify numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in categorical_columns and col != 'Age']
    
    print("\nNumeric columns:", numeric_columns)
    
    # Handle numeric columns
    numeric_imputer = SimpleImputer(strategy='median')
    data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])
    
    # Handle categorical columns
    for col in categorical_columns:
        if col in data.columns:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            data[[col]] = categorical_imputer.fit_transform(data[[col]])
    
    # Apply StandardScaler to numeric columns (except Age)
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    print("\nFinal data types:")
    print(data.dtypes)
    
    return data

def prepare_data() -> Tuple[pd.DataFrame, List[str], Dict[str, List]]:
    try:
        behavioral, hcp = load_data()
        print("Data loaded successfully")
        
        behavioral, hcp = select_features(behavioral, hcp)
        print("Features selected successfully")
        
        data = pd.merge(hcp, behavioral, on="Subject")
        print("Data merged successfully")
        print(f"Merged data shape: {data.shape}")
        
        categorical_columns = ['Gender', 'MMSE_Score', 'Age']
        
        processed_data = preprocess_data(data, categorical_columns)
        print("Data preprocessed successfully")
        
        # Get categories for categorical variables
        categories = {}
        for col in categorical_columns:
            if col in processed_data.columns:
                categories[col] = sorted(processed_data[col].dropna().unique().tolist())
        
        return processed_data, categorical_columns, categories
    except Exception as e:
        print(f"An error occurred in prepare_data: {str(e)}")
        raise