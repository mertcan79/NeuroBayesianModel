import numpy as np
import pandas as pd
from scipy import stats

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

def analyze_age_related_changes(data, age_column, cognitive_measures, age_groups=None):
    """
    Analyze age-related changes in cognitive measures.

    Args:
        data (pd.DataFrame): Input data.
        age_column (str): Name of the column containing age information.
        cognitive_measures (list): List of cognitive measures to analyze.
        age_groups (dict, optional): Dictionary of age groups and their ranges.

    Returns:
        dict: Analysis results for each cognitive measure.
    """
    if age_groups is None:
        age_groups = {
            'Young': (0, 30),
            'Middle': (31, 60),
            'Old': (61, 100)
        }

    results = {}
    for measure in cognitive_measures:
        measure_results = {}
        for group, (min_age, max_age) in age_groups.items():
            group_data = data[(data[age_column] >= min_age) & (data[age_column] <= max_age)]
            measure_results[group] = {
                'mean': group_data[measure].mean(),
                'std': group_data[measure].std(),
                'correlation_with_age': stats.pearsonr(group_data[age_column], group_data[measure])[0]
            }
        
        # Perform ANOVA to test for significant differences between age groups
        f_value, p_value = stats.f_oneway(*[data[(data[age_column] >= min_age) & (data[age_column] <= max_age)][measure] 
                                            for _, (min_age, max_age) in age_groups.items()])
        measure_results['anova'] = {'f_value': f_value, 'p_value': p_value}
        
        results[measure] = measure_results

    return results

def perform_interaction_effects_analysis(data, target_variable, predictors, interaction_terms):
    """
    Analyze interaction effects between predictors on the target variable.

    Args:
        data (pd.DataFrame): Input data.
        target_variable (str): Name of the target variable.
        predictors (list): List of predictor variables.
        interaction_terms (list): List of tuples representing interaction terms.

    Returns:
        pd.DataFrame: Summary of interaction effects.
    """
    import statsmodels.formula.api as smf

    # Prepare the formula
    main_effects = ' + '.join(predictors)
    interactions = ' + '.join([f'{a}:{b}' for a, b in interaction_terms])
    formula = f"{target_variable} ~ {main_effects} + {interactions}"

    # Fit the model
    model = smf.ols(formula=formula, data=data).fit()

    # Extract interaction effects
    interaction_effects = model.summary2().tables[1].loc[[f'{a}:{b}' for a, b in interaction_terms]]

    return interaction_effects