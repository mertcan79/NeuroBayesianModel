import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer

# Load data
data = pd.read_csv('data/connectome_data.csv')

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
model = BayesianNetwork([
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
])

# Fit the model
model.fit(df, estimator=BayesianEstimator, prior_type="BDeu")

# Create an inference object
inference = VariableElimination(model)

# Example: Predict CogFluidComp_Unadj given some evidence
evidence = {
    'Age': 2,  # Corresponds to '31-35' age range
    'Gender': 1,  # Assuming '1' represents one gender category
    'MMSE_Score': 3,  # Using 3 instead of 4, as we have 5 bins (0-4)
}

prediction = inference.query(['CogFluidComp_Unadj'], evidence=evidence)
print("Predicted distribution for CogFluidComp_Unadj:")
print(prediction)

# Example: Compute probability of high cognitive function given evidence
high_cog = inference.query(['CogFluidComp_Unadj', 'CogCrystalComp_Unadj'], 
                           evidence={'Age': 2, 'MMSE_Score': 3})
print("\nProbability of high cognitive function:")
print(high_cog)

# Analyze feature importance through Markov blanket
for node in ['CogFluidComp_Unadj', 'CogCrystalComp_Unadj']:
    print(f"\nMarkov Blanket for {node}:")
    print(model.get_markov_blanket(node))