import numpy as np
import pandas as pd

def perform_age_stratified_analysis(data, age_column, target_variable, age_groups):
    """
    Perform age-stratified analysis on the data.

    Args:
        data (pd.DataFrame): Input data.
        age_column (str): Column containing age information.
        target_variable (str): The target variable to analyze.
        age_groups (dict): Dictionary of age groups and their corresponding ranges.

    Returns:
        dict: Dictionary containing the results of the analysis.
    """
    results = {}
    for group, (min_age, max_age) in age_groups.items():
        group_data = data[
            (data[age_column] >= min_age) & (data[age_column] <= max_age)
        ]
        results[group] = {
            "correlation": group_data.corr()[target_variable].to_dict(),
            "mean": group_data.mean().to_dict(),
            "std": group_data.std().to_dict(),
        }
    return results

def analyze_brain_cognitive_correlations(data, brain_features, cognitive_features):
    """
    Analyze the correlations between brain features and cognitive features.

    Args:
        data (pd.DataFrame): Input data.
        brain_features (list): List of brain features.
        cognitive_features (list): List of cognitive features.

    Returns:
        pd.Series: Series containing the correlations between brain features and cognitive features.
    """
    brain_data = data[brain_features]
    cognitive_data = data[cognitive_features]
    return brain_data.corrwith(cognitive_data, method="pearson").dropna()
