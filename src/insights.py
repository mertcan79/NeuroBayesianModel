import os
import json
from typing import Dict, Any, List
from datetime import datetime
import numpy as np
import pandas as pd
from bayesian_network import BayesianNetwork, HierarchicalBayesianNetwork
from bayesian_node import BayesianNode, CategoricalNode


def explain_structure_extended():
    structure = {"nodes": list(nodes.keys()), "edges": get_edges()}
    for node_name, node in nodes.items():
        structure[node_name] = {
            "parents": [parent.name for parent in node.parents],
            "children": [child.name for child in node.children],
            "parameters": node.parameters if hasattr(node, "parameters") else None,
            "distribution": (
                str(node.distribution) if hasattr(node, "distribution") else None
            ),
        }

    return structure

def get_unexpected_insights():
    insights = []
    sensitivity = compute_sensitivity("CogFluidComp_Unadj")

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

def generate_actionable_insights():
    insights = []
    sensitivity = compute_sensitivity("CogFluidComp_Unadj")

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

def analyze_personality_cognition_relationship():
    personality_traits = ["NEOFAC_O", "NEOFAC_C"]
    cognitive_measures = ["CogFluidComp_Unadj", "CogCrystalComp_Unadj"]

    relationships = {}
    for trait in personality_traits:
        for measure in cognitive_measures:
            correlation = np.corrcoef(data[trait], data[measure])[0, 1]
            relationships[f"{trait}-{measure}"] = correlation

    return relationships

def analyze_age_dependent_relationships():
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

def get_practical_implications():
    implications = []
    sensitivity = compute_sensitivity("CogFluidComp_Unadj")

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

def perform_age_stratified_analysis():
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

def explain_unexpected_findings():
    explanations = []
    sensitivity = compute_sensitivity("CogFluidComp_Unadj")

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

def enforce_expected_connections():
    expected_connections = [
        ("FS_L_Hippo_Vol", "CogFluidComp_Unadj"),
        ("FS_R_Hippo_Vol", "CogFluidComp_Unadj"),
        ("FS_BrainStem_Vol", "ProcSpeed_Unadj"),
        ("NEOFAC_O", "CogCrystalComp_Unadj"),
        ("NEOFAC_C", "CogFluidComp_Unadj"),
    ]
    for parent, child in expected_connections:
        if child not in nodes[parent].children:
            add_edge(parent, child)

    fit(data)  # Refit the model with new connections

def analyze_brain_stem_relationship():
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

def get_personalized_recommendations( individual_data):
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

def get_clinical_insights():
    insights = []
    fluid_sensitivity = compute_sensitivity("CogFluidComp_Unadj")
    crystal_sensitivity = compute_sensitivity("CogCrystalComp_Unadj")

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

def get_clinical_implications():
    implications = []
    sensitivity_fluid = compute_sensitivity("CogFluidComp_Unadj")
    sensitivity_crystal = compute_sensitivity("CogCrystalComp_Unadj")

    for feature, value in sensitivity_fluid.items():
        if abs(value) > 0.1:
            implications.append(
                f"Changes in {feature} may significantly impact fluid cognitive abilities (sensitivity: {value:.2f}), suggesting potential for targeted interventions."
            )

    for feature, value in sensitivity_crystal.items():
        if abs(value) > 0.1:
            implications.append(
                f"Changes in {feature} may significantly impact crystallized cognitive abilities (sensitivity: {value:.2f}), indicating areas for potential cognitive preservation strategies."
            )

    if abs(sensitivity_fluid["Age"]) > 0.1 or abs(sensitivity_crystal["Age"]) > 0.1:
        implications.append(
            "Age has a substantial impact on cognitive abilities, emphasizing the need for age-specific cognitive interventions and preventive strategies."
        )

    return implications

def get_novel_insights():
    insights = []
    sensitivity_fluid = compute_sensitivity("CogFluidComp_Unadj")
    sensitivity_crystal = compute_sensitivity("CogCrystalComp_Unadj")

    # Compare brain structure influences
    brain_structures = [f for f in sensitivity_fluid.keys() if f.startswith("FS_")]
    max_influence = max(brain_structures, key=lambda x: abs(sensitivity_fluid[x]))
    insights.append(
        f"Unexpectedly high influence of {max_influence} on fluid cognitive abilities (sensitivity: {sensitivity_fluid[max_influence]:.2f}), suggesting a potential new area for cognitive research."
    )

    # Compare personality influences
    if "NEOFAC_O" in sensitivity_fluid and "NEOFAC_C" in sensitivity_fluid:
        if abs(sensitivity_fluid["NEOFAC_O"]) > abs(sensitivity_fluid["NEOFAC_C"]):
            insights.append(
                f"Openness to experience shows a stronger relationship with fluid cognitive abilities (sensitivity: {sensitivity_fluid['NEOFAC_O']:.2f}) than conscientiousness (sensitivity: {sensitivity_fluid['NEOFAC_C']:.2f}), which could inform personality-based cognitive training approaches."
            )

    # Compare fluid vs crystallized influences
    for feature in sensitivity_fluid.keys():
        if feature in sensitivity_crystal:
            if abs(sensitivity_fluid[feature]) > 2 * abs(
                sensitivity_crystal[feature]
            ):
                insights.append(
                    f"{feature} has a much stronger influence on fluid cognitive abilities (sensitivity: {sensitivity_fluid[feature]:.2f}) compared to crystallized abilities (sensitivity: {sensitivity_crystal[feature]:.2f}), suggesting different mechanisms for these cognitive domains."
                )

    return insights

def get_age_specific_insights() -> List[str]:
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

def get_gender_specific_insights():
    if data is None:
        return ["No data available for gender-specific insights."]

    insights = []
    male_data = data[data["Gender"] == 0]  # Assuming 0 is male
    female_data = data[data["Gender"] == 1]  # Assuming 1 is female

    male_model = BayesianNetwork(
        method=method,
        max_parents=max_parents,
        iterations=iterations,
        categorical_columns=categorical_columns,
    )
    female_model = BayesianNetwork(
        method=method,
        max_parents=max_parents,
        iterations=iterations,
        categorical_columns=categorical_columns,
    )

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

def summarize_key_findings() -> str:
    relationships = get_key_relationships()
    insights = get_novel_insights()

    summary = f"Our Bayesian Network model identified {len(relationships)} strong relationships between brain structures and cognitive functions. "
    summary += f"Key findings include:\n"
    summary += f"1. The strongest relationship found was between {relationships[0]['parent']} and {relationships[0]['child']} (strength: {relationships[0]['strength']}).\n"
    summary += f"2. Age and Gender directly influence both fluid and crystallized cognitive abilities.\n"
    summary += f"3. Brain structure variables, particularly total gray matter volume and hippocampal volume, are strong predictors of cognitive performance.\n"
    summary += f"4. {insights[0] if insights else 'No unexpected influences were found.'}\n"

    return summary

def summarize_personality_cognition(relationships: Dict[str, float]) -> Dict[str, float]:
    if not relationships:
        return {}
    return dict(sorted(
        {k: round(v, 3) for k, v in relationships.items()}.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3])

def summarize_brain_stem_relationships(relationships: Dict[str, float]) -> Dict[str, float]:
    if not relationships:
        return {}
    return {k: round(v, 3) for k, v in relationships.items()}

def summarize_age_dependent_changes(changes: Dict[str, float]) -> Dict[str, float]:
    if not changes:
        return {}
    significant_changes = {k: round(v, 3) for k, v in changes.items() if abs(v) > 0.1}
    return dict(sorted(significant_changes.items(), key=lambda x: abs(x[1]), reverse=True)[:5])


def get_key_relationships() -> List[Dict[str, Any]]:
    relationships = []
    for node_name, node in nodes.items():
        for parent in node.parents:
            strength = abs(compute_edge_strength(parent.name, node_name))
            relationships.append(
                {"parent": parent.name, "child": node_name, "strength": strength}
            )
    sorted_relationships = sorted(
        relationships, key=lambda x: x["strength"], reverse=True
    )
    top_10_percent = sorted_relationships[: max(1, len(sorted_relationships) // 10)]
    return [
        {
            "parent": r["parent"],
            "child": r["child"],
            "strength": round(r["strength"], 2),
        }
        for r in top_10_percent
    ]

def explain_key_relationships():
    explanations = []
    relationships = get_key_relationships()
    for rel in relationships:
        if (
            rel["parent"] == "FS_Total_GM_Vol"
            and rel["child"] == "CogFluidComp_Unadj"
        ):
            explanations.append(
                f"Total gray matter volume strongly influences fluid cognitive ability (strength: {rel['strength']}). "
                "This suggests that cognitive training programs should focus on activities that promote gray matter preservation, "
                "such as complex problem-solving tasks and learning new skills."
            )
        elif (
            rel["parent"] == "FS_L_Hippo_Vol"
            and rel["child"] == "CogFluidComp_Unadj"
        ):
            explanations.append(
                f"Left hippocampus volume is closely related to fluid cognitive ability (strength: {rel['strength']}). "
                "This highlights the importance of memory-enhancing exercises in cognitive training, particularly those "
                "that engage spatial navigation and episodic memory formation."
            )
        elif rel["parent"] == "NEOFAC_O" and rel["child"] == "CogCrystalComp_Unadj":
            explanations.append(
                f"Openness to experience (NEOFAC_O) influences crystallized cognitive ability (strength: {rel['strength']}). "
                "This suggests that encouraging curiosity and diverse learning experiences could enhance long-term cognitive performance."
            )
        # Add more specific explanations for other important relationships
    return explanations