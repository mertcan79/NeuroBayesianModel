import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    behavioral = pd.read_csv('/Users/macbookair/Documents/neurobayesianmodel/data/connectome_behavioral.csv')
    hcp = pd.read_csv('/Users/macbookair/Documents/neurobayesianmodel/data/hcp_freesurfer.csv')
    return behavioral, hcp

def select_features(behavioral: pd.DataFrame, hcp: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    relevant_features_behavioral = [
        'Subject', 'Age', 'Gender', 'CogFluidComp_Unadj', 'CogCrystalComp_Unadj', 'MMSE_Score',
        'NEOFAC_O', 'NEOFAC_C', 'ProcSpeed_Unadj', 'CardSort_Unadj', 'PicVocab_Unadj', 'ReadEng_Unadj'
    ]

    relevant_features_hcp = [
        'Subject', 'Gender', 'FS_TotCort_GM_Vol', 'FS_SubCort_GM_Vol', 'FS_Total_GM_Vol', 'FS_Tot_WM_Vol', 'FS_BrainStem_Vol',
        'FS_L_Hippo_Vol', 'FS_R_Hippo_Vol', 'FS_L_Amygdala_Vol', 'FS_R_Amygdala_Vol',
        'FS_L_Caudate_Vol', 'FS_R_Caudate_Vol', 'FS_L_Putamen_Vol', 'FS_R_Putamen_Vol',
    ]

    relevant_features_hcp_temporal = [col for col in hcp.columns if 'temporal' in col.lower()]
    relevant_features_hcp = list(set(relevant_features_hcp + relevant_features_hcp_temporal))

    hcp = hcp[relevant_features_hcp].copy()
    behavioral = behavioral[relevant_features_behavioral].copy()

    return behavioral, hcp

def preprocess_data(data: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    data = data.copy()
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in categorical_columns]

    numeric_imputer = SimpleImputer(strategy='median')
    data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])

    categorical_imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])

    for col in categorical_columns:
        data[col] = pd.Categorical(data[col]).codes

    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    return data

def prepare_data() -> Tuple[pd.DataFrame, List[str]]:
    behavioral, hcp = load_data()
    behavioral, hcp = select_features(behavioral, hcp)
    data = pd.merge(hcp, behavioral, on=["Subject", "Gender"])
    
    categorical_columns = ['Gender', 'Age', 'MMSE_Score']
    
    processed_data = preprocess_data(data, categorical_columns)
    
    return processed_data, categorical_columns