import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from bayesian_network import BayesianNetwork

# Load data
data = pd.read_csv('/Users/macbookair/Documents/bayesian_dataclass/data/connectome_behavioral.csv')

# Select a smaller subset of relevant features
relevant_features = [
    'Age', 'Gender', 'CogFluidComp_Unadj', 'CogCrystalComp_Unadj', 'MMSE_Score'
]

df = data[relevant_features].copy()

# Handle categorical variables
df['Age'] = pd.Categorical(df['Age']).codes
df['Gender'] = pd.Categorical(df['Gender']).codes

# Handle missing data
imputer = SimpleImputer(strategy='median')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Define a simpler Bayesian Network structure
prior_edges = [
    ('Age', 'CogFluidComp_Unadj'),
    ('Age', 'CogCrystalComp_Unadj'),
    ('Gender', 'CogFluidComp_Unadj'),
    ('Gender', 'CogCrystalComp_Unadj'),
    ('MMSE_Score', 'CogFluidComp_Unadj'),
    ('MMSE_Score', 'CogCrystalComp_Unadj')
]

# Create and fit the model
model = BayesianNetwork(method='hill_climb', max_parents=2)
model.fit(df, prior_edges=prior_edges)

# Compute and print log-likelihood
log_likelihood = model.log_likelihood(df)
print(f"Log-likelihood: {log_likelihood}")

# Perform cross-validation
mean_ll, std_ll = model.cross_validate(df, k_folds=5)
print(f"Cross-validation: Mean LL = {mean_ll:.4f}, Std = {std_ll:.4f}")

# Compute sensitivity for CogFluidComp_Unadj
sensitivity = model.compute_sensitivity('CogFluidComp_Unadj', num_samples=1000)
top_sensitivities = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
print("\nTop sensitivities for CogFluidComp_Unadj:")
for node, value in top_sensitivities:
    print(f"Sensitivity to {node}: {value:.4f}")

# Print the network structure
print("\nNetwork structure:")
print(model.explain_structure())