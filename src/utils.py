import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.feature_selection import mutual_info_regression

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

def write_results_to_json(network, data: pd.DataFrame, results: Dict[str, Any], filename: str = None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"
    
    log_folder = "logs"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    file_path = os.path.join(log_folder, filename)

    # Add all analysis results
    results["network_structure"] = network.explain_structure_extended()
    results["edge_probabilities"] = network.compute_edge_probabilities()
    results["key_relationships"] = network.get_key_relationships()
    results["feature_importance"] = compute_feature_importance(network, data)
    results["unexpected_insights"] = get_unexpected_insights(network)
    results["actionable_insights"] = generate_actionable_insights(network)
    results["personality_cognition_relationships"] = analyze_personality_cognition_relationship(data)
    results["age_dependent_relationships"] = analyze_age_dependent_relationships(data)
    results["practical_implications"] = get_practical_implications(network)
    results["age_stratified_analysis"] = perform_age_stratified_analysis(data)
    results["unexpected_findings_explanations"] = explain_unexpected_findings(network)
    results["brain_stem_relationships"] = analyze_brain_stem_relationship(data)
    results["clinical_insights"] = get_clinical_insights(network)
    results["age_specific_insights"] = get_age_specific_insights(data)
    results["gender_specific_insights"] = get_gender_specific_insights(data)
    results["key_findings_summary"] = summarize_key_findings(network)
    results["marginal_likelihoods"] = network.compute_marginal_likelihoods()
    results["influential_nodes"] = network.identify_influential_nodes()
    results["mutual_information"] = network.compute_mutual_information()
    results["node_influence"] = network.compute_node_influence()
    results["pairwise_mutual_information"] = network.compute_pairwise_mutual_information()
    results["interaction_effects"] = perform_interaction_effects_analysis(data, network.target)
    results["counterfactual_analysis"] = perform_counterfactual_analysis(network, data, {'NEOFAC_O': 5})
    results["mediation_analysis"] = perform_mediation_analysis(data, 'NEOFAC_O', 'ProcSpeed_Unadj', 'CogFluidComp_Unadj')
    results["sensitivity_analysis"] = perform_sensitivity_analysis(network, data, network.target)
    results["vif"] = compute_vif(data)
    results["outliers"] = detect_outliers(data)
    results["correlations"] = compute_correlations(data)
    results["partial_correlations"] = compute_partial_correlations(data)
    results["heteroscedasticity"] = perform_heteroscedasticity_test(data, network.target)

    serializable_results = make_serializable(results)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Results successfully written to {file_path}")
    except Exception as e:
        print(f"Error writing results to file: {str(e)}")

def write_summary_to_json(network, results: Dict[str, Any], filename: str = None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bayesian_network_summary_{timestamp}.json"
    
    log_folder = "logs"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    file_path = os.path.join(log_folder, filename)

    summary = {
        "network_structure": network.explain_structure_extended(),
        "mean_log_likelihood": results.get("mean_log_likelihood"),
        "std_log_likelihood": results.get("std_log_likelihood"),
        "sensitivity": results.get("sensitivity"),
        "num_nodes": len(network.nodes),
        "num_edges": len(network.get_edges()),
        "categorical_variables": network.categorical_columns,
        "continuous_variables": [node for node in network.nodes if node not in network.categorical_columns],
        "key_findings": summarize_key_findings(network),
        "future_research_directions": network.suggest_future_research(),
        "key_personality_cognition_findings": summarize_personality_cognition(results.get("personality_cognition_relationships")),
        "significant_age_dependent_changes": summarize_age_dependent_changes(results.get("age_dependent_relationships")),
    }

    try:
        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary successfully written to {file_path}")
    except Exception as e:
        print(f"Error writing summary to file: {str(e)}")

# Utility functions
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

def compute_feature_importance(model, data: pd.DataFrame) -> Dict[str, float]:
    X = data.drop(columns=[model.target])
    y = data[model.target]
    mi_scores = mutual_info_regression(X, y)
    return dict(zip(X.columns, mi_scores))

def get_unexpected_insights(network):
    insights = []
    sensitivity = network.compute_sensitivity("CogFluidComp_Unadj")
    
    if sensitivity["FS_R_Amygdala_Vol"] > sensitivity["FS_L_Amygdala_Vol"]:
        insights.append("Right amygdala volume shows stronger influence on fluid cognitive ability than left amygdala volume.")
    
    if sensitivity["NEOFAC_O"] > sensitivity["NEOFAC_C"]:
        insights.append("Openness to experience has a stronger relationship with fluid cognitive ability than conscientiousness.")
    
    if sensitivity["FS_Tot_WM_Vol"] > sensitivity["FS_Total_GM_Vol"]:
        insights.append("Total white matter volume shows stronger influence on cognitive performance than total gray matter volume.")
    
    return insights

def generate_actionable_insights(network):
    insights = []
    sensitivity = network.compute_sensitivity("CogFluidComp_Unadj")
    
    if sensitivity["FS_L_Hippo_Vol"] > 0.1:
        insights.append("Focus on memory exercises to potentially improve fluid cognitive abilities.")
    
    if sensitivity["NEOFAC_O"] > 0.1:
        insights.append("Encourage openness to new experiences as part of cognitive training.")
    
    if sensitivity["FS_BrainStem_Vol"] > 0.1:
        insights.append("Consider incorporating balance and coordination exercises to potentially benefit cognitive function.")
    
    return insights

def analyze_personality_cognition_relationship(data: pd.DataFrame):
    personality_traits = ["NEOFAC_O", "NEOFAC_C"]
    cognitive_measures = ["CogFluidComp_Unadj", "CogCrystalComp_Unadj"]
    
    relationships = {}
    for trait in personality_traits:
        for measure in cognitive_measures:
            correlation = np.corrcoef(data[trait], data[measure])[0, 1]
            relationships[f"{trait}-{measure}"] = correlation
    
    return relationships

def analyze_age_dependent_relationships(data: pd.DataFrame):
    young_data = data[data["Age"] < data["Age"].median()]
    old_data = data[data["Age"] >= data["Age"].median()]
    
    young_correlations = young_data.corr()["CogFluidComp_Unadj"]
    old_correlations = old_data.corr()["CogFluidComp_Unadj"]
    
    age_differences = {}
    for feature in young_correlations.index:
        age_differences[feature] = old_correlations[feature] - young_correlations[feature]
    
    return age_differences

def get_practical_implications(network):
    implications = []
    sensitivity = network.compute_sensitivity("CogFluidComp_Unadj")
    
    if sensitivity["FS_Total_GM_Vol"] > 0.1:
        implications.append("Focus on exercises that promote gray matter preservation, such as learning new skills or languages.")
    
    if sensitivity["FS_Tot_WM_Vol"] > 0.1:
        implications.append("Incorporate tasks that challenge white matter integrity, like complex problem-solving or strategic thinking exercises.")
    
    if sensitivity["FS_BrainStem_Vol"] > 0.1:
        implications.append("Consider including balance and coordination exercises, which may indirectly benefit cognitive function through brain stem activation.")
    
    if sensitivity["NEOFAC_O"] > 0.1:
        implications.append("Encourage openness to new experiences as part of the cognitive training regimen.")
    
    return implications

def perform_age_stratified_analysis(data: pd.DataFrame):
    age_groups = {"Young": (0, 30), "Middle": (31, 60), "Older": (61, 100)}
    
    results = {}
    for group, (min_age, max_age) in age_groups.items():
        group_data = data[(data["Age"] >= min_age) & (data["Age"] <= max_age)]
        results[group] = {
            "correlation": group_data.corr()["CogFluidComp_Unadj"].to_dict(),
            "mean": group_data.mean().to_dict(),
            "std": group_data.std().to_dict()
        }
    
    return results

def explain_unexpected_findings(network):
    explanations = []
    sensitivity = network.compute_sensitivity("CogFluidComp_Unadj")
    
    if sensitivity["FS_BrainStem_Vol"] < -0.1:
        explanations.append("Unexpectedly, brain stem volume shows a negative relationship with fluid cognitive ability. This could suggest complex compensatory mechanisms.")
    
    if sensitivity["FS_R_Amygdala_Vol"] > sensitivity["FS_L_Amygdala_Vol"]:
        explanations.append("The right amygdala volume appears to have a stronger influence on fluid cognitive ability than the left. This asymmetry might indicate a more significant role of right-hemisphere emotional processing in cognitive flexibility.")
    
    return explanations

def analyze_brain_stem_relationship(data: pd.DataFrame):
    brain_stem_correlations = {}
    for measure in ["CogFluidComp_Unadj", "CogCrystalComp_Unadj", "ProcSpeed_Unadj"]:
        correlation = np.corrcoef(data["FS_BrainStem_Vol"], data[measure])[0, 1]
        brain_stem_correlations[measure] = correlation
    return brain_stem_correlations

def get_clinical_insights(network):
    insights = []
    sensitivity = network.compute_sensitivity("CogFluidComp_Unadj")
    
    for feature, value in sensitivity.items():
        if abs(value) > 0.1:
            insights.append(f"{feature} has a significant impact on fluid cognitive abilities (sensitivity: {value:.2f})")
    
    return insights

def get_age_specific_insights(data: pd.DataFrame) -> List[str]:
    young_data = data[data["Age"] < data["Age"].median()]
    old_data = data[data["Age"] >= data["Age"].median()]
    
    young_corr = young_data.corr()["CogFluidComp_Unadj"]
    old_corr = old_data.corr()["CogFluidComp_Unadj"]
    
    insights = []
    for feature in young_corr.index:
        if abs(young_corr[feature] - old_corr[feature]) > 0.1:
            if young_corr[feature] > old_corr[feature]:
                insights.append(f"{feature} has a stronger influence on fluid cognitive abilities in younger individuals")
            else:
                insights.append(f"{feature} has a stronger influence on fluid cognitive abilities in older individuals")
    
    return insights

def get_gender_specific_insights(data: pd.DataFrame):
    male_data = data[data["Gender"] == 0]
    female_data = data[data["Gender"] == 1]
    
    male_corr = male_data.corr()["CogFluidComp_Unadj"]
    female_corr = female_data.corr()["CogFluidComp_Unadj"]
    
    insights = []
    for feature in male_corr.index:
        if abs(male_corr[feature] - female_corr[feature]) > 0.1:
            if male_corr[feature] > female_corr[feature]:
                insights.append(f"{feature} has a stronger influence on fluid cognitive abilities in males")
            else:
                insights.append(f"{feature} has a stronger influence on fluid cognitive abilities in females")
    
    return insights

def summarize_key_findings(network) -> str:
    summary = []
    sensitivity = network.compute_sensitivity("CogFluidComp_Unadj")
    top_features = sorted(sensitivity.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    summary.append("Top 5 influential features for fluid cognitive ability:")
    for feature, value in top_features:
        summary.append(f"- {feature}: {value:.2f}")
    
    return "\n".join(summary)

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

def summarize_personality_cognition(relationships: Dict[str, float]) -> str:
    summary = "Personality-Cognition Relationships:\n"
    for relationship, correlation in relationships.items():
        trait, measure = relationship.split('-')
        strength = "strong" if abs(correlation) > 0.5 else "moderate" if abs(correlation) > 0.3 else "weak"
        direction = "positive" if correlation > 0 else "negative"
        summary += f"- {trait} shows a {strength} {direction} relationship with {measure} (r = {correlation:.2f})\n"
    return summary

def summarize_age_dependent_changes(age_differences: Dict[str, float]) -> str:
    summary = "Age-Dependent Changes in Cognitive Relationships:\n"
    for variable, difference in age_differences.items():
        if abs(difference) > 0.1:
            direction = "stronger" if difference > 0 else "weaker"
            summary += f"- The relationship between {variable} and cognitive ability becomes {direction} with age (Δr = {difference:.2f})\n"
    return summary