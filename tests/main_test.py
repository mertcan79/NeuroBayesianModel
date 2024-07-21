import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats
from bayesian_network import BayesianNetwork
from typing import List
import json

# Load data
behavioral = pd.read_csv('/Users/macbookair/Documents/bayesian_dataclass/data/connectome_behavioral.csv')
hcp = pd.read_csv('/Users/macbookair/Documents/bayesian_dataclass/data/hcp_freesurfer.csv')

# Select relevant features
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
relevant_features_hcp = list(set(relevant_features_hcp.copy() + relevant_features_hcp_temporal))
relevant_features_hcp

hcp = hcp[relevant_features_hcp].copy()
behavioral = behavioral[relevant_features_behavioral].copy()

data = pd.merge(hcp, behavioral, on=["Subject", "Gender"])


def preprocess_data(df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    df = df.copy()

    # Handle 'Age' column separately
    if 'Age' in df.columns:
        df['Age'] = pd.Categorical(df['Age']).codes

    # Convert other categorical columns to numeric codes
    for col in categorical_columns:
        if col != 'Age':
            df[col] = df[col].astype('category').cat.codes

    # Separate into categorical and numeric columns
    numeric_features = [col for col in df.columns if col not in categorical_columns]

    # Handle missing data for numeric features
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])

    # Scale numeric features
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    return df

# Specify categorical columns
categorical_columns = ['Age', 'Gender']

df_processed = preprocess_data(data, categorical_columns)

# Define Bayesian Network structure with more specific connections
prior_edges = [
    ('Age', 'CogFluidComp_Unadj'),
    ('Age', 'CogCrystalComp_Unadj'),
    ('Age', 'MMSE_Score'),
    ('Gender', 'CogFluidComp_Unadj'),
    ('Gender', 'CogCrystalComp_Unadj'),
    ('MMSE_Score', 'CogFluidComp_Unadj'),
    ('MMSE_Score', 'CogCrystalComp_Unadj'),
    ('FS_Total_GM_Vol', 'CogFluidComp_Unadj'),
    ('FS_Total_GM_Vol', 'CogCrystalComp_Unadj'),
    ('FS_Tot_WM_Vol', 'CogFluidComp_Unadj'),
    ('FS_Tot_WM_Vol', 'CogCrystalComp_Unadj'),
    ('FS_L_Hippo_Vol', 'CogFluidComp_Unadj'),
    ('FS_R_Hippo_Vol', 'CogFluidComp_Unadj'),
    ('FS_L_Amygdala_Vol', 'NEOFAC_O'),
    ('FS_R_Amygdala_Vol', 'NEOFAC_O'),
    ('FS_L_Caudate_Vol', 'ProcSpeed_Unadj'),
    ('FS_R_Caudate_Vol', 'ProcSpeed_Unadj'),
    ('NEOFAC_O', 'CogCrystalComp_Unadj'),
    ('NEOFAC_C', 'CogFluidComp_Unadj'),
    ('CogFluidComp_Unadj', 'ProcSpeed_Unadj'),
    ('CogFluidComp_Unadj', 'CardSort_Unadj'),
    ('CogCrystalComp_Unadj', 'PicVocab_Unadj'),
    ('CogCrystalComp_Unadj', 'ReadEng_Unadj'),
    # New connections based on neuroanatomical knowledge
    ('FS_TotCort_GM_Vol', 'CogCrystalComp_Unadj'),
    ('FS_BrainStem_Vol', 'ProcSpeed_Unadj'),
    ('FS_L_Putamen_Vol', 'CardSort_Unadj'),
    ('FS_R_Putamen_Vol', 'CardSort_Unadj'),
    ('FS_L_Hippo_Vol', 'NEOFAC_O'),
    ('FS_R_Hippo_Vol', 'NEOFAC_O'),
]

# Add connections for temporal lobe measures
temporal_measures = [col for col in df_processed.columns if 'temporal' in col.lower()]
for measure in temporal_measures:
    prior_edges.append((measure, 'CogCrystalComp_Unadj'))
    prior_edges.append((measure, 'ReadEng_Unadj'))

# Create and fit the model
model = BayesianNetwork(method='hill_climb', max_parents=5)  # Increased max_parents
model.fit(df_processed, prior_edges=prior_edges)

# Compute and print log-likelihood
log_likelihood = model.log_likelihood(df_processed)
print(f"Log-likelihood: {log_likelihood}")

# Perform cross-validation
mean_ll, std_ll = model.cross_validate(df_processed, k_folds=5)
print(f"Cross-validation: Mean LL = {mean_ll:.4f}, Std = {std_ll:.4f}")

# Compute sensitivity for CogFluidComp_Unadj
sensitivity = model.compute_sensitivity('CogFluidComp_Unadj', num_samples=1000)
top_sensitivities = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 sensitivities for CogFluidComp_Unadj:")
for node, value in top_sensitivities:
    print(f"Sensitivity to {node}: {value:.4f}")

# Print the network structure
print("\nNetwork structure:")
print(model.explain_structure())

# Perform Metropolis-Hastings sampling
observed_data = {
    'Age': 0,  # Assuming 0 represents the median age after coding
    'Gender': 1,  # Assuming 1 represents one gender category
    'MMSE_Score': 0,  # Assuming 0 represents the median MMSE score after StandardScaler
}
mh_samples = model.metropolis_hastings(observed_data, num_samples=1000)

print("\nMetropolis-Hastings sampling results:")
for node, samples in mh_samples.items():
    if node not in observed_data:
        print(f"{node}: Mean = {np.mean(samples):.4f}, Std = {np.std(samples):.4f}")

print("\nAnalysis complete.")