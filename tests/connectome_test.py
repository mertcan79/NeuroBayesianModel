import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
from bayesian_network import BayesianNetwork

# Load data
data = pd.read_csv('/Users/macbookair/Documents/bayesian_dataclass/data/connectome_behavioral.csv')

# Select relevant features
relevant_features = [
    'Age', 'Gender', 'FS_IntraCranial_Vol', 'FS_BrainSeg_Vol',
    'CogFluidComp_Unadj', 'CogCrystalComp_Unadj', 'MMSE_Score',
    'NEOFAC_N', 'NEOFAC_E', 'NEOFAC_O', 'NEOFAC_A', 'NEOFAC_C',
    'PicSeq_Unadj', 'CardSort_Unadj', 'Flanker_Unadj', 'ReadEng_Unadj',
    'PicVocab_Unadj', 'ProcSpeed_Unadj', 'ListSort_Unadj'
]

df = data[relevant_features]

# Handle categorical variables
df['Age'] = pd.Categorical(df['Age']).codes
df['Gender'] = pd.Categorical(df['Gender']).codes

# Handle missing data
numeric_features = df.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='median')
df[numeric_features] = imputer.fit_transform(df[numeric_features])

# Discretize continuous variables
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
continuous_vars = df.select_dtypes(include=['float64']).columns
df[continuous_vars] = discretizer.fit_transform(df[continuous_vars])

# Convert all columns to integers
for col in df.columns:
    df[col] = df[col].astype(int)

# Define Bayesian Network structure
prior_edges = [
    ('Age', 'CogFluidComp_Unadj'),
    ('Age', 'CogCrystalComp_Unadj'),
    ('Gender', 'CogFluidComp_Unadj'),
    ('Gender', 'CogCrystalComp_Unadj'),
    ('FS_IntraCranial_Vol', 'CogFluidComp_Unadj'),
    ('FS_BrainSeg_Vol', 'CogFluidComp_Unadj'),
    ('MMSE_Score', 'CogFluidComp_Unadj'),
    ('MMSE_Score', 'CogCrystalComp_Unadj'),
    ('NEOFAC_N', 'CogFluidComp_Unadj'),
    ('NEOFAC_E', 'CogFluidComp_Unadj'),
    ('NEOFAC_O', 'CogCrystalComp_Unadj'),
    ('NEOFAC_A', 'CogCrystalComp_Unadj'),
    ('NEOFAC_C', 'CogFluidComp_Unadj'),
    ('CogFluidComp_Unadj', 'PicSeq_Unadj'),
    ('CogFluidComp_Unadj', 'CardSort_Unadj'),
    ('CogFluidComp_Unadj', 'Flanker_Unadj'),
    ('CogCrystalComp_Unadj', 'ReadEng_Unadj'),
    ('CogCrystalComp_Unadj', 'PicVocab_Unadj'),
    ('CogFluidComp_Unadj', 'ProcSpeed_Unadj'),
    ('CogFluidComp_Unadj', 'ListSort_Unadj')
]

# Create and fit the model
model = BayesianNetwork(method='hill_climb', max_parents=5)
model.fit(df, prior_edges=prior_edges)

# Compute and print log-likelihood
log_likelihood = model.log_likelihood(df)
print(f"Log-likelihood: {log_likelihood}")

# Perform cross-validation
mean_ll, std_ll = model.cross_validate(df, k_folds=5)
print(f"Cross-validation: Mean LL = {mean_ll:.4f}, Std = {std_ll:.4f}")

# Compute sensitivity for CogFluidComp_Unadj
sensitivity = model.compute_sensitivity('CogFluidComp_Unadj', num_samples=1000)
top_sensitivities = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop 5 sensitivities for CogFluidComp_Unadj:")
for node, value in top_sensitivities:
    print(f"Sensitivity to {node}: {value:.4f}")

# Perform Metropolis-Hastings sampling
observed_data = {
    'Age': 2,  # Corresponds to '31-35' age range
    'Gender': 1,  # Assuming '1' represents one gender category
    'MMSE_Score': 3,  # Using 3 instead of 4, as we have 5 bins (0-4)
}
mh_samples = model.metropolis_hastings(observed_data, num_samples=1000)

print("\nMetropolis-Hastings sampling results:")
for node, samples in mh_samples.items():
    if node not in observed_data:
        print(f"{node}: Mean = {np.mean(samples):.4f}, Std = {np.std(samples):.4f}")

# Print the network structure
print("\nNetwork structure:")
print(model.explain_structure())