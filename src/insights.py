from typing import Dict, Any, List
import numpy as np
import pandas as pd
from bayesian_network import BayesianNetwork

def get_unexpected_insights(network: BayesianNetwork):
    insights = []
    sensitivity = network.compute_sensitivity(target_node="CogFluidComp_Unadj")

    if sensitivity["FS_R_Amygdala_Vol"] > sensitivity["FS_L_Amygdala_Vol"]:
        insights.append(
            "Right amygdala volume shows a stronger influence on fluid cognitive ability than left amygdala volume. "
            "This unexpected finding suggests that emotional processing in the right hemisphere might play a larger role in cognitive flexibility. "
            "Consider incorporating emotional regulation techniques, particularly those targeting right hemisphere processing, into cognitive training programs."
        )

    if sensitivity["NEOFAC_O"] > sensitivity["NEOFAC_C"]:
        insights.append(
            "Openness to experience (NEOFAC_O) has a stronger relationship with fluid cognitive ability than conscientiousness (NEOFAC_C). "
            "This counter-intuitive result suggests that encouraging exploration and creativity in cognitive training programs "
            "might be more beneficial than strictly structured approaches. Consider designing adaptive, open-ended problem-solving tasks."
        )

    if sensitivity["FS_Tot_WM_Vol"] > sensitivity["FS_Total_GM_Vol"]:
        insights.append(
            "Total white matter volume shows a stronger influence on cognitive performance than total gray matter volume. "
            "This unexpected finding highlights the importance of connectivity between brain regions. "
            "Consider incorporating exercises that promote white matter integrity, such as complex motor skill learning or meditation practices."
        )

    return insights

def generate_actionable_insights(network: BayesianNetwork):
    insights = []
    sensitivity = network.compute_sensitivity(target_node="CogFluidComp_Unadj")

    if sensitivity["FS_L_Hippo_Vol"] > 0.1:
        insights.append(
            "Focus on memory exercises to potentially improve fluid cognitive abilities."
        )

    if sensitivity["NEOFAC_O"] > 0.1:
        insights.append(
            "Encourage openness to new experiences as part of cognitive training."
        )

    if sensitivity["FS_BrainStem_Vol"] > 0.1:
        insights.append(
            "Consider incorporating balance and coordination exercises to potentially benefit cognitive function."
        )

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
    young_data = data[data["Age"] < 30]
    old_data = data[data["Age"] >= 30]

    young_model = BayesianNetwork()
    old_model = BayesianNetwork()

    young_model.fit(young_data)
    old_model.fit(old_data)

    young_sensitivity = young_model.compute_sensitivity("CogFluidComp_Unadj")
    old_sensitivity = old_model.compute_sensitivity("CogFluidComp_Unadj")

    age_differences = {}
    for key in young_sensitivity.keys():
        age_differences[key] = old_sensitivity[key] - young_sensitivity[key]

    return age_differences

def get_practical_implications(network: BayesianNetwork):
    implications = []
    sensitivity = network.compute_sensitivity(network, "CogFluidComp_Unadj")

    if sensitivity["FS_Total_GM_Vol"] > 0.1:
        implications.append(
            "Focus on exercises that promote gray matter preservation, such as learning new skills or languages."
        )

    if sensitivity["FS_Tot_WM_Vol"] > 0.1:
        implications.append(
            "Incorporate tasks that challenge white matter integrity, like complex problem-solving or strategic thinking exercises."
        )

    if sensitivity["FS_BrainStem_Vol"] > 0.1:
        implications.append(
            "Consider including balance and coordination exercises, which may indirectly benefit cognitive function through brain stem activation."
        )

    if sensitivity["NEOFAC_O"] > 0.1:
        implications.append(
            "Encourage openness to new experiences as part of the cognitive training regimen."
        )

    return implications

def perform_age_stratified_analysis(data: pd.DataFrame):
    age_groups = {"Young": (0, 30), "Middle": (31, 60), "Older": (61, 100)}

    results = {}
    for group, (min_age, max_age) in age_groups.items():
        group_data = data[
            (data["Age"] >= min_age) & (data["Age"] <= max_age)
        ]
        group_model = BayesianNetwork()
        group_model.fit(group_data)
        results[group] = {
            "sensitivity": group_model.compute_sensitivity("CogFluidComp_Unadj"),
            "key_relationships": group_model.get_key_relationships(),
        }

    return results

def explain_unexpected_findings(network: BayesianNetwork):
    explanations = []
    sensitivity = network.compute_sensitivity("CogFluidComp_Unadj")

    if sensitivity["FS_BrainStem_Vol"] < -0.1:
        explanations.append(
            "Unexpectedly, brain stem volume shows a negative relationship with fluid cognitive ability. "
            "This could suggest that the brain stem's role in cognition is more complex than previously thought, "
            "possibly involving compensatory mechanisms or reflecting broader neurodevelopmental processes."
        )

    if sensitivity["FS_R_Amygdala_Vol"] > sensitivity["FS_L_Amygdala_Vol"]:
        explanations.append(
            "The right amygdala volume appears to have a stronger influence on fluid cognitive ability than the left. "
            "This asymmetry might indicate a more significant role of right-hemisphere emotional processing in cognitive flexibility."
        )

    return explanations

def analyze_brain_stem_relationship(data: pd.DataFrame):
    brain_stem_correlations = {}
    for measure in [
        "CogFluidComp_Unadj",
        "CogCrystalComp_Unadj",
        "ProcSpeed_Unadj",
    ]:
        correlation = np.corrcoef(
            data["FS_BrainStem_Vol"], data[measure]
        )[0, 1]
        brain_stem_correlations[measure] = correlation
    return brain_stem_correlations

def get_personalized_recommendations(data: pd.DataFrame, individual_data):
    recommendations = []

    if individual_data["FS_Total_GM_Vol"] < data["FS_Total_GM_Vol"].mean():
        recommendations.append(
            "Focus on activities that promote gray matter preservation, such as learning a new language or musical instrument."
        )

    if individual_data["FS_Tot_WM_Vol"] < data["FS_Tot_WM_Vol"].mean():
        recommendations.append(
            "Engage in tasks that challenge white matter integrity, like complex problem-solving or strategic games."
        )

    if individual_data["NEOFAC_O"] > data["NEOFAC_O"].mean():
        recommendations.append(
            "Leverage your openness to experience with diverse and novel cognitive challenges."
        )

    return recommendations

def get_clinical_insights(network: BayesianNetwork):
    insights = []
    fluid_sensitivity = network.compute_sensitivity("CogFluidComp_Unadj")
    crystal_sensitivity = network.compute_sensitivity("CogCrystalComp_Unadj")

    for feature, value in fluid_sensitivity.items():
        if abs(value) > 0.1:
            insights.append(
                f"{feature} has a significant impact on fluid cognitive abilities (sensitivity: {value:.2f})"
            )

    for feature, value in crystal_sensitivity.items():
        if abs(value) > 0.1:
            insights.append(
                f"{feature} has a significant impact on crystallized cognitive abilities (sensitivity: {value:.2f})"
            )

    return insights

def get_age_specific_insights(data: pd.DataFrame) -> List[str]:
    if data is None:
        return ["No data available for age-specific insights."]

    insights = []
    young_data = data[data["Age"] < data["Age"].median()]
    old_data = data[data["Age"] >= data["Age"].median()]

    young_model = BayesianNetwork()
    old_model = BayesianNetwork()

    young_model.fit(young_data)
    old_model.fit(old_data)

    young_sensitivity = young_model.compute_sensitivity("CogFluidComp_Unadj")
    old_sensitivity = old_model.compute_sensitivity("CogFluidComp_Unadj")

    for feature in young_sensitivity.keys():
        if abs(young_sensitivity[feature] - old_sensitivity[feature]) > 0.1:
            if young_sensitivity[feature] > old_sensitivity[feature]:
                insights.append(
                    f"{feature} has a stronger influence on fluid cognitive abilities in younger individuals "
                    f"(sensitivity difference: {young_sensitivity[feature] - old_sensitivity[feature]:.2f})"
                )
            else:
                insights.append(
                    f"{feature} has a stronger influence on fluid cognitive abilities in older individuals "
                    f"(sensitivity difference: {old_sensitivity[feature] - young_sensitivity[feature]:.2f})"
                )

    return insights

def get_gender_specific_insights(data: pd.DataFrame):
    if data is None:
        return ["No data available for gender-specific insights."]

    insights = []
    male_data = data[data["Gender"] == 0]  # Assuming 0 is male
    female_data = data[data["Gender"] == 1]  # Assuming 1 is female

    male_model = BayesianNetwork()
    female_model = BayesianNetwork()

    male_model.fit(male_data)
    female_model.fit(female_data)

    male_sensitivity = male_model.compute_sensitivity("CogFluidComp_Unadj")
    female_sensitivity = female_model.compute_sensitivity("CogFluidComp_Unadj")

    for feature in male_sensitivity.keys():
        if abs(male_sensitivity[feature] - female_sensitivity[feature]) > 0.1:
            if male_sensitivity[feature] > female_sensitivity[feature]:
                insights.append(
                    f"{feature} has a stronger influence on fluid cognitive abilities in males (sensitivity difference: {male_sensitivity[feature] - female_sensitivity[feature]:.2f})"
                )
            else:
                insights.append(
                    f"{feature} has a stronger influence on fluid cognitive abilities in females (sensitivity difference: {female_sensitivity[feature] - male_sensitivity[feature]:.2f})"
                )

    return insights

def summarize_key_findings(network: BayesianNetwork) -> str:
    return network.summarize_key_findings()

def get_key_relationships(network: BayesianNetwork) -> List[Dict[str, Any]]:
    return network.get_key_relationships()

def compute_marginal_likelihoods(network: BayesianNetwork) -> str:
    return network.compute_marginal_likelihoods()

def compute_edge_probabilities(network: BayesianNetwork) -> str:
    return network.compute_edge_probabilities()

def identify_influential_nodes(network: BayesianNetwork) -> str:
    return network.identify_influential_nodes()

def compute_mutual_information(network: BayesianNetwork) -> str:
    return network.compute_mutual_information()

def compute_edge_probability(network: BayesianNetwork) -> str:
    return network.compute_edge_probability()

def compute_node_influence(network: BayesianNetwork) -> str:
    return network.compute_node_influence()

def compute_pairwise_mutual_information(network: BayesianNetwork) -> str:
    return network.compute_pairwise_mutual_information()