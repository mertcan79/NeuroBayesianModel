import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats
from bayesian_network import BayesianNetwork, HierarchicalBayesianNetwork
from typing import List
import logging

logging.basicConfig(level=logging.INFO)

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

levels = ["low", "mid", "high"]
level_constraints = {
        "mid": ["low"],
        "high": ["low", "mid"]
    }

# Include all FreeSurfer variables, especially those with "temporal" or "Temporal"
relevant_features_hcp_temporal = [col for col in hcp.columns if 'temporal' in col.lower()]
relevant_features_hcp = list(set(relevant_features_hcp.copy() + relevant_features_hcp_temporal))

hcp = hcp[relevant_features_hcp].copy()
behavioral = behavioral[relevant_features_behavioral].copy()

data = pd.merge(hcp, behavioral, on=["Subject", "Gender"])

# Specify categorical columns
categorical_columns = ['Gender', 'Age', 'MMSE_Score']

def preprocess_data(data: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    data = data.copy()
    
    # Handle missing values
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist() + ['MMSE_Score']
    
    numeric_imputer = SimpleImputer(strategy='median')
    data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])
    
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])
    
    # Encode categorical variables
    for col in categorical_columns:
        data[col] = pd.Categorical(data[col]).codes
    
    # Scale numeric features
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    return data

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
    ('FS_TotCort_GM_Vol', 'CogCrystalComp_Unadj'),
    ('FS_BrainStem_Vol', 'ProcSpeed_Unadj'),
    ('FS_L_Putamen_Vol', 'CardSort_Unadj'),
    ('FS_R_Putamen_Vol', 'CardSort_Unadj'),
    ('FS_L_Hippo_Vol', 'NEOFAC_O'),
    ('FS_R_Hippo_Vol', 'NEOFAC_O'),
]

# Add connections for temporal lobe measures
temporal_measures = [col for col in data.columns if 'temporal' in col.lower()]
for measure in temporal_measures:
    prior_edges.append((measure, 'CogCrystalComp_Unadj'))
    prior_edges.append((measure, 'ReadEng_Unadj'))

# Filter prior_edges to only include edges between existing columns
existing_columns = set(df_processed.columns)
filtered_prior_edges = [
    (parent, child) for parent, child in prior_edges
    if parent in existing_columns and child in existing_columns
]

print("Filtered prior edges:")
print(filtered_prior_edges)

# Debugging: Print nodes in prior edges
print("Nodes in prior edges:")
all_nodes_in_edges = set([node for edge in prior_edges for node in edge])
print(all_nodes_in_edges)

# Check if 'Gender' exists in the dataframe columns
print("Columns in df_processed:")
print(df_processed.columns)

# When creating the model
model = BayesianNetwork(method='hill_climb', max_parents=5, categorical_columns=categorical_columns)

# When fitting the model
model.fit(df_processed, prior_edges=filtered_prior_edges)

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
network_structure = model.explain_structure_extended()
print(network_structure)

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

# Write results to JSON
results = {
    "log_likelihood": log_likelihood,
    "cross_validation": {"mean": mean_ll, "std": std_ll},
    "top_sensitivities": dict(top_sensitivities),
    "network_structure": network_structure,
    "mh_samples": {node: {"mean": float(np.mean(samples)), "std": float(np.std(samples))} 
                   for node, samples in mh_samples.items() if node not in observed_data}
}
model.write_results_to_json(results)

print("\nAnalysis complete.")

# Example of using HierarchicalBayesianNetwork
hierarchical_levels = ['cellular', 'regional', 'functional']
h_model = HierarchicalBayesianNetwork(levels=hierarchical_levels, method='hill_climb', max_parents=3)

# Add nodes to specific levels
h_model.add_node('FS_L_Hippo_Vol', 'regional')
h_model.add_node('CogFluidComp_Unadj', 'functional')
h_model.add_node('NEOFAC_O', 'functional')

# Add edges
h_model.add_edge('FS_L_Hippo_Vol', 'CogFluidComp_Unadj')
h_model.add_edge('FS_L_Hippo_Vol', 'NEOFAC_O')

# Fit the hierarchical model
h_model.fit(df_processed, level_constraints=level_constraints)

# Print hierarchical structure
print("\nHierarchical Network Structure:")
print(h_model.explain_structure_extended())
print("\nHierarchical Analysis complete.")
