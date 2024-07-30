import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats


def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    else:
        return obj

def calculate_results(network, data: pd.DataFrame, params: Dict[str, Any]) -> Dict:

    results = {}
    try:
        results["edge_probabilities"] = network.compute_all_edge_probabilities()
        results["network_structure"] = network.explain_structure_extended()
        results["edge_probabilities"] = network.compute_all_edge_probabilities()
        results["key_relationships"] = network.get_key_relationships()
        results["feature_importance"] = compute_feature_importance(data, params["target_variable"])
        results["unexpected_insights"] = get_unexpected_insights(network, params["target_variable"], params["analysis_variables"])
        results["actionable_insights"] = generate_actionable_insights(network, data, params["target_variable"])
        results["personality_cognition_relationships"] = analyze_personality_cognition_relationship(data, params["personality_traits"], params["cognitive_measures"])
        results["age_dependent_relationships"] = analyze_age_dependent_relationships(data, params["age_column"], params["target_variable"])
        results["practical_implications"] = get_practical_implications(network, params["target_variable"], params["feature_thresholds"])
        results["age_stratified_analysis"] = perform_age_stratified_analysis(data, params["age_column"], params["target_variable"], params["age_groups"])
        results["unexpected_findings_explanations"] = explain_unexpected_findings(network, params["target_variable"], params["brain_stem_column"], "FS_L_Amygdala_Vol", "FS_R_Amygdala_Vol")
        results["brain_stem_relationships"] = analyze_brain_stem_relationship(data, params["brain_stem_column"], params["cognitive_measures"])
        results["clinical_insights"] = get_clinical_insights(network, params["target_variable"], {"Brain Structure": params["brain_structure_features"], "Personality": params["personality_traits"]})
        results["age_specific_insights"] = get_age_specific_insights(data, params["age_column"], params["target_variable"])
        results["gender_specific_insights"] = get_gender_specific_insights(data, params["gender_column"], params["target_variable"])
        results["key_findings_summary"] = summarize_key_findings(network, params["target_variable"])
        results["network_structure"] = network.explain_structure_extended()
        results["summarize_personality_cognition"] = summarize_personality_cognition(results["personality_cognition_relationships"])
        results["mean_log_likelihood"] = get_mean_log_likelihood(network, data)
        results["std_log_likelihood"] = get_std_log_likelihood(network, data)
        results["generate_comprehensive_insights"] = generate_comprehensive_insights(network, data, params["target_variable"])

    except Exception as e:
        print(f"Error in calculate_results: {str(e)}")
        print(f"Params: {params}")
        raise

    return results

def write_results_to_json(results: Dict, filename: str = None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
    
    log_folder = "logs"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    file_path = os.path.join(log_folder, filename)

    serializable_results = make_serializable(results)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Results successfully written to {file_path}")
    except Exception as e:
        print(f"Error writing results to file: {str(e)}")

# Utility functions

def get_mean_log_likelihood(network, data):
    log_likelihoods = [network.compute_log_likelihood(sample) for _, sample in data.iterrows()]
    return np.mean(log_likelihoods)

def get_std_log_likelihood(network, data):
    log_likelihoods = [network.compute_log_likelihood(sample) for _, sample in data.iterrows()]
    return np.std(log_likelihoods)

def compute_vif(data: pd.DataFrame) -> Dict[str, float]:
    X = data.select_dtypes(include=[np.number])
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.set_index("Variable")["VIF"].to_dict()

def detect_outliers(data: pd.DataFrame, threshold: float = 3) -> Dict[str, List[int]]:
    outliers = {}
    for column in data.select_dtypes(include=[np.number]):
        z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
        outliers[column] = list(data.index[z_scores > threshold])
    return outliers

def compute_correlations(data: pd.DataFrame) -> pd.DataFrame:
    return data.corr()

def compute_partial_correlations(data: pd.DataFrame) -> pd.DataFrame:
    return data.pcorr()

def perform_heteroscedasticity_test(data: pd.DataFrame, target: str) -> Dict[str, float]:
    X = data.drop(columns=[target])
    y = data[target]
    _, pvalue, _, _ = het_breuschpagan(y, X)
    return {"p_value": pvalue}

def compute_feature_importance(data: pd.DataFrame, target_variable: str) -> Dict[str, float]:
    from sklearn.feature_selection import mutual_info_regression
    
    X = data.drop(columns=[target_variable])
    y = data[target_variable]
    mi_scores = mutual_info_regression(X, y)
    return dict(zip(X.columns, mi_scores))

def get_unexpected_insights(network, target_variable, analysis_variables):
    insights = []
    sensitivity = network.compute_sensitivity(target_variable)
    # Analyze hemispheric differences
    for left, right in analysis_variables:
        if left in sensitivity and right in sensitivity:
            diff = sensitivity[right] - sensitivity[left]
            if abs(diff) > 0.1:
                insights.append(f"{right} shows {'stronger' if diff > 0 else 'weaker'} influence on {target_variable} than {left} (difference: {diff:.2f}). This suggests potential hemispheric specialization.")
    
    # Analyze personality traits
    personality_traits = [trait for trait in sensitivity if trait.startswith('NEOFAC_')]
    if len(personality_traits) >= 2:
        top_trait = max(personality_traits, key=lambda x: abs(sensitivity[x]))
        for trait in personality_traits:
            if trait != top_trait:
                diff = sensitivity[top_trait] - sensitivity[trait]
                if abs(diff) > 0.1:
                    insights.append(f"{top_trait} has a stronger relationship with {target_variable} than {trait} (difference: {diff:.2f}). This may indicate that {top_trait.split('_')[1]} is more closely tied to {target_variable}.")
    
    # Analyze brain structure relationships
    brain_structures = [struct for struct in sensitivity if struct.startswith('FS_')]
    if len(brain_structures) >= 2:
        top_structure = max(brain_structures, key=lambda x: abs(sensitivity[x]))
        insights.append(f"{top_structure} shows the strongest influence on {target_variable} among brain structures (sensitivity: {sensitivity[top_structure]:.2f}). This suggests its critical role in cognitive performance.")
    
    return insights

def generate_actionable_insights(network, data, target_variable):
    if not target_variable:
        raise ValueError("Target variable is empty")
    
    insights = []
    # Identify top influencers
    sensitivities = network.compute_sensitivity(target_variable)
    top_influencers = sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    for feature, sensitivity in top_influencers:
        if feature.startswith('FS_'):
            median = data[feature].median()
            below_median = data[data[feature] < median][target_variable].mean()
            above_median = data[data[feature] >= median][target_variable].mean()
            diff = above_median - below_median
            insights.append(f"Increasing {feature} from below to above median is associated with a {diff:.2f} change in {target_variable}")
        elif feature.startswith('NEOFAC_'):
            trait = feature.split('_')[1]
            insights.append(f"Focus on improving {trait} trait, as it's strongly associated with {target_variable}")
        elif feature in ['ProcSpeed_Unadj', 'CardSort_Unadj']:
            insights.append(f"Cognitive training focusing on {feature} may improve overall {target_variable}")
    
    return insights

def analyze_personality_cognition_relationship(data, personality_traits, cognitive_measures) -> Dict:
    relationships = {}
    for trait in personality_traits:
        for measure in cognitive_measures:
            correlation, p_value = stats.pearsonr(data[trait], data[measure])
            relationships[f"{trait}-{measure}"] = {
                "correlation": correlation,
                "p_value": p_value,
                "significance": "Significant" if p_value < 0.05 else "Not Significant"
            }
    
    # Find the strongest relationship
    strongest = max(relationships.items(), key=lambda x: abs(x[1]["correlation"]))
    
    analysis = {
        "relationships": relationships,
        "strongest_relationship": {
            "traits": strongest[0],
            "correlation": strongest[1]["correlation"],
            "p_value": strongest[1]["p_value"]
        },
        "interpretation": f"The strongest relationship is between {strongest[0]} (r={strongest[1]['correlation']:.2f}, p={strongest[1]['p_value']:.4f}). This suggests that {strongest[0].split('-')[0]} is particularly important for {strongest[0].split('-')[1]}."
    }
    
    return analysis

def analyze_age_dependent_relationships(data: pd.DataFrame, age_column: str, target_variable: str):
    median_age = data[age_column].median()
    young_data = data[data[age_column] < median_age]
    old_data = data[data[age_column] >= median_age]
    
    young_correlations = young_data.corr()[target_variable]
    old_correlations = old_data.corr()[target_variable]
    
    age_differences = {}
    for feature in young_correlations.index:
        if feature != target_variable:
            age_differences[feature] = old_correlations[feature] - young_correlations[feature]
    
    return age_differences

def get_practical_implications(network, target_variable: str, feature_thresholds: Dict[str, float]):
    implications = []
    sensitivity = network.compute_sensitivity(target_variable)
    
    for feature, threshold in feature_thresholds.items():
        if sensitivity[feature] > threshold:
            if feature.startswith("FS_"):
                implications.append(f"Focus on exercises that promote {feature} preservation or enhancement.")
            elif feature.startswith("NEOFAC_"):
                trait = feature.split('_')[1]
                implications.append(f"Encourage {trait} as part of the cognitive training regimen.")
    
    return implications

def perform_age_stratified_analysis(data: pd.DataFrame, age_column: str, target_variable: str, age_groups: Dict[str, Tuple[int, int]]):
    results = {}
    for group, (min_age, max_age) in age_groups.items():
        group_data = data[(data[age_column] >= min_age) & (data[age_column] <= max_age)]
        results[group] = {
            "correlation": group_data.corr()[target_variable].to_dict(),
            "mean": group_data.mean().to_dict(),
            "std": group_data.std().to_dict()
        }
    
    return results

def explain_unexpected_findings(network, target_variable: str, brain_stem_column: str, left_amygdala_column: str, right_amygdala_column: str, threshold: float = -0.1):
    explanations = []
    sensitivity = network.compute_sensitivity(target_variable)
    
    if sensitivity[brain_stem_column] < threshold:
        explanations.append(f"Unexpectedly, {brain_stem_column} shows a negative relationship with {target_variable}. This could suggest complex compensatory mechanisms.")
    
    if sensitivity[right_amygdala_column] > sensitivity[left_amygdala_column]:
        explanations.append(f"The {right_amygdala_column} appears to have a stronger influence on {target_variable} than the {left_amygdala_column}. This asymmetry might indicate a more significant role of right-hemisphere emotional processing in cognitive flexibility.")
    
    return explanations

def analyze_brain_stem_relationship(data: pd.DataFrame, brain_stem_column: str, target_variables: List[str]):
    brain_stem_correlations = {}
    for measure in target_variables:
        correlation = np.corrcoef(data[brain_stem_column], data[measure])[0, 1]
        brain_stem_correlations[measure] = correlation
    return brain_stem_correlations

def get_clinical_insights(network, target_variable, feature_categories):
    insights = []
    sensitivity = network.compute_sensitivity(target_variable)
    
    # Analyze top influences in each category
    for category, features in feature_categories.items():
        category_features = [f for f in sensitivity if f in features]
        if category_features:
            top_feature = max(category_features, key=lambda x: abs(sensitivity[x]))
            value = sensitivity[top_feature]
            insights.append(f"The most influential {category} feature for {target_variable} is {top_feature} (sensitivity: {value:.2f}). ")
            
            if category == "Brain Structure":
                insights[-1] += f"This suggests that {'increases' if value > 0 else 'decreases'} in {top_feature} are associated with {'higher' if value > 0 else 'lower'} {target_variable}."
            elif category == "Personality":
                insights[-1] += f"This indicates that the personality trait of {top_feature.split('_')[1]} plays a significant role in {target_variable}."
            else:
                insights[-1] += f"This highlights the importance of {top_feature} in overall cognitive function."
    
    # Analyze interactions
    edge_probabilities = network.compute_all_edge_probabilities()
    if edge_probabilities:
        strongest_edge = max(edge_probabilities.items(), key=lambda x: x[1])
        insights.append(f"The strongest relationship in the network is between {strongest_edge[0][0]} and {strongest_edge[0][1]} (probability: {strongest_edge[1]:.2f}).")
    
    return insights


def get_age_specific_insights(data: pd.DataFrame, age_column: str, target_variable: str, threshold: float = 0.1) -> List[str]:
    median_age = data[age_column].median()
    young_data = data[data[age_column] < median_age]
    old_data = data[data[age_column] >= median_age]
    
    young_corr = young_data.corr()[target_variable]
    old_corr = old_data.corr()[target_variable]
    
    insights = []
    for feature in young_corr.index:
        if feature != target_variable and abs(young_corr[feature] - old_corr[feature]) > threshold:
            if young_corr[feature] > old_corr[feature]:
                insights.append(f"{feature} has a stronger influence on {target_variable} in younger individuals")
            else:
                insights.append(f"{feature} has a stronger influence on {target_variable} in older individuals")
    
    return insights

def get_gender_specific_insights(data: pd.DataFrame, gender_column: str, target_variable: str, male_value: int = 0, female_value: int = 1, threshold: float = 0.1):
    male_data = data[data[gender_column] == male_value]
    female_data = data[data[gender_column] == female_value]
    
    male_corr = male_data.corr()[target_variable]
    female_corr = female_data.corr()[target_variable]
    
    insights = []
    for feature in male_corr.index:
        if feature != target_variable and abs(male_corr[feature] - female_corr[feature]) > threshold:
            if male_corr[feature] > female_corr[feature]:
                insights.append(f"{feature} has a stronger influence on {target_variable} in males")
            else:
                insights.append(f"{feature} has a stronger influence on {target_variable} in females")
    
    return insights

def summarize_key_findings(network, target_variable: str, top_n: int = 5) -> str:
    sensitivity = network.compute_sensitivity(target_variable)
    top_features = sorted(sensitivity.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    
    summary = f"Top {top_n} influential features for {target_variable}:\n"
    for feature, value in top_features:
        summary += f"- {feature}: {value:.2f}\n"
    
    return summary

def perform_interaction_effects_analysis(data: pd.DataFrame, target: str):
    interactions = {}
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for i in range(len(numeric_columns)):
        for j in range(i+1, len(numeric_columns)):
            col1, col2 = numeric_columns[i], numeric_columns[j]
            interaction_term = data[col1] * data[col2]
            correlation = np.corrcoef(interaction_term, data[target])[0, 1]
            interactions[f"{col1}_{col2}"] = correlation
    return interactions

def perform_counterfactual_analysis(network, data: pd.DataFrame, interventions: Dict[str, float]):
    original_prediction = network.predict(data)
    counterfactual_data = data.copy()
    for variable, value in interventions.items():
        counterfactual_data[variable] = value
    counterfactual_prediction = network.predict(counterfactual_data)
    return {
        "original_prediction": original_prediction.mean(),
        "counterfactual_prediction": counterfactual_prediction.mean(),
        "difference": (counterfactual_prediction - original_prediction).mean()
    }

def perform_mediation_analysis(data: pd.DataFrame, independent: str, mediator: str, dependent: str):
    c = np.corrcoef(data[independent], data[dependent])[0, 1]
    a = np.corrcoef(data[independent], data[mediator])[0, 1]
    b = np.corrcoef(data[mediator], data[dependent])[0, 1]
    c_prime = c - (a * b)
    return {
        "total_effect": c,
        "indirect_effect": a * b,
        "direct_effect": c_prime
    }

def perform_sensitivity_analysis(network, data: pd.DataFrame, target: str, perturbation: float = 0.1):
    results = {}
    for column in data.columns:
        if column != target:
            perturbed_data = data.copy()
            perturbed_data[column] *= (1 + perturbation)
            original_prediction = network.predict(data)
            perturbed_prediction = network.predict(perturbed_data)
            sensitivity = np.mean(np.abs(perturbed_prediction - original_prediction))
            results[column] = sensitivity
    return results

def summarize_personality_cognition(personality_cognition_relationships):
    if not personality_cognition_relationships:
        return "No personality-cognition relationships found."
    
    summary = []
    relationships = personality_cognition_relationships.get("relationships", {})
    for relationship, details in relationships.items():
        trait, measure = relationship.split('-')
        correlation = details['correlation']
        p_value = details['p_value']
        summary.append(f"{trait} shows relationship with {measure} (r={correlation:.2f}, p={p_value:.4f})")
    
    if summary:
        strongest_relationship = max(relationships.items(), key=lambda x: abs(x[1]['correlation']))
        summary.append(f"\nStrongest relationship: {strongest_relationship[0]} (r={strongest_relationship[1]['correlation']:.2f}, p={strongest_relationship[1]['p_value']:.4f})")
    
    return "\n".join(summary)

def summarize_age_dependent_changes(age_differences: Dict[str, float]) -> str:
    summary = "Age-Dependent Changes in Cognitive Relationships:\n"
    for variable, difference in age_differences.items():
        if abs(difference) > 0.1:
            direction = "stronger" if difference > 0 else "weaker"
            summary += f"- The relationship between {variable} and cognitive ability becomes {direction} with age (Δr = {difference:.2f})\n"
    return summary

def generate_comprehensive_insights(network, data, target_variable):
    insights = []
    
    # Check for non-linear relationships
    for column in data.columns:
        if column != target_variable:
            correlation = np.corrcoef(data[column], data[target_variable])[0,1]
            mutual_info = network.compute_mutual_information(column, target_variable)
            if abs(correlation) < 0.1 and mutual_info > 0.1:
                insights.append(f"Potential non-linear relationship between {column} and {target_variable}")
    
    # Check for interaction effects
    for col1 in data.columns:
        for col2 in data.columns:
            if col1 != col2 and col1 != target_variable and col2 != target_variable:
                interaction = data[col1] * data[col2]
                int_corr = np.corrcoef(interaction, data[target_variable])[0,1]
                if abs(int_corr) > 0.2:
                    insights.append(f"Potential interaction effect between {col1} and {col2} on {target_variable}")
    
    # Check for threshold effects
    for column in data.columns:
        if column != target_variable:
            quantiles = data[column].quantile([0.25, 0.75])
            low_corr = np.corrcoef(data[data[column] < quantiles[0.25]][column], 
                                   data[data[column] < quantiles[0.25]][target_variable])[0,1]
            high_corr = np.corrcoef(data[data[column] > quantiles[0.75]][column], 
                                    data[data[column] > quantiles[0.75]][target_variable])[0,1]
            if abs(high_corr - low_corr) > 0.2:
                insights.append(f"Potential threshold effect for {column} on {target_variable}")
    
    return insights