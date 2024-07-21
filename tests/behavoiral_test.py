import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from scipy import stats
from bayesian_network import BayesianNetwork
from typing import List

# Load data
data = pd.read_csv('/Users/macbookair/Documents/bayesian_dataclass/data/connectome_behavioral.csv')

# Select relevant features
relevant_features = [
    'Age', 'Gender', 'FS_BrainSeg_Vol',
    'CogFluidComp_Unadj', 'CogCrystalComp_Unadj', 'MMSE_Score',
    'NEOFAC_O', 'NEOFAC_C',
    'ProcSpeed_Unadj', 'CardSort_Unadj', 'PicVocab_Unadj', 'ReadEng_Unadj'
]

df = data[relevant_features].copy()

def preprocess_data(df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    df = df.copy()

    # Handle 'Age' column separately
    if 'Age' in df.columns:
        df['Age'] = pd.Categorical(df['Age']).codes

    # Convert other categorical columns to numeric codes
    for col in categorical_columns:
        if col != 'Age':  # Skip 'Age' as we've already handled it
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

df_processed = preprocess_data(df, categorical_columns)

# The rest of your code remains the same
def assign_distributions(df):
    distributions = {}
    for column in df.columns:
        data = df[column].dropna()
        if data.dtype == 'object' or data.nunique() < 10:
            # Categorical or discrete data
            distributions[column] = stats.rv_discrete(values=(range(data.nunique()), data.value_counts(normalize=True).sort_index()))
        elif column == 'Age':
            # Age is now treated as a discrete variable
            distributions[column] = stats.rv_discrete(values=(range(data.nunique()), data.value_counts(normalize=True).sort_index()))
        else:
            # Continuous data, test for normality
            _, p_value = stats.normaltest(data)
            if p_value > 0.05:
                distributions[column] = stats.norm
            else:
                distributions[column] = stats.gamma  # or another appropriate distribution

    return distributions

distributions = assign_distributions(df_processed)

# Define Bayesian Network structure
prior_edges = [
    ('Age', 'CogFluidComp_Unadj'),
    ('Age', 'CogCrystalComp_Unadj'),
    ('Age', 'MMSE_Score'),
    ('Gender', 'CogFluidComp_Unadj'),
    ('Gender', 'CogCrystalComp_Unadj'),
    ('MMSE_Score', 'CogFluidComp_Unadj'),
    ('MMSE_Score', 'CogCrystalComp_Unadj'),
    ('FS_BrainSeg_Vol', 'CogFluidComp_Unadj'),
    ('NEOFAC_O', 'CogCrystalComp_Unadj'),
    ('NEOFAC_C', 'CogFluidComp_Unadj'),
    ('CogFluidComp_Unadj', 'ProcSpeed_Unadj'),
    ('CogFluidComp_Unadj', 'CardSort_Unadj'),
    ('CogCrystalComp_Unadj', 'PicVocab_Unadj'),
    ('CogCrystalComp_Unadj', 'ReadEng_Unadj'),
]

# Create and fit the model
model = BayesianNetwork(method='hill_climb', max_parents=3)
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
    'Age': 0,  # Assuming 0 represents the median age after RobustScaler
    'Gender_1': 1,  # Assuming 1 represents one gender category
    'MMSE_Score': 0,  # Assuming 0 represents the median MMSE score after StandardScaler
}
mh_samples = model.metropolis_hastings(observed_data, num_samples=1000)

print("\nMetropolis-Hastings sampling results:")
for node, samples in mh_samples.items():
    if node not in observed_data:
        print(f"{node}: Mean = {np.mean(samples):.4f}, Std = {np.std(samples):.4f}")

print("\nAnalysis complete.")
