import os
from loguru import logger
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from src.data_processing import prepare_data
import warnings
import jax
from src.tasks import (fit_model, compute_edge_weights, explain_structure, get_key_relationships, compute_sensitivities, 
                       bayesian_model_comparison, cross_validate, get_clinical_implications, analyze_age_dependent_relationships, 
                       perform_interaction_effects_analysis, perform_counterfactual_analysis, analyze_age_related_changes, 
                       perform_sensitivity_analysis, fit_nonlinear_model, compare_performance
                       )
                        
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

logger.add("logs/hbn_{time}.log", rotation="500 MB")

environment = os.getenv("ENVIRONMENT", "local")
data_path = (
    os.getenv("LOCAL_DATA_PATH")
    if environment == "local"
    else os.getenv("CLOUD_DATA_PATH")
)
processed_data_path = (
    os.getenv("LOCAL_DATA_PATH_PROCESSED")
    if environment == "local"
    else os.getenv("CLOUD_DATA_PATH")
)

behavioral_path = os.path.join(data_path, "connectome_behavioral.csv")
behavioral_path_processed = os.path.join(
    processed_data_path, "connectome_behavioral.csv"
)
hcp_path = os.path.join(data_path, "hcp_freesurfer.csv")
hcp_path_processed = os.path.join(processed_data_path, "hcp_freesurfer.csv")


def preprocess_hcp():
    def map_age_to_category(age_str):
        bins = ["22-25", "26-30", "31-35", "36+"]
        categories = [1, 2, 3, 4]
        if pd.isna(age_str):
            return np.nan
        age_str = age_str.strip()
        return categories[bins.index(age_str)] if age_str in bins else np.nan

    def process_age_gender(data: pd.DataFrame) -> pd.DataFrame:
        if "Age" in data.columns and data["Age"].dtype == "object":
            data["Age"] = data["Age"].apply(map_age_to_category)
        return data

    behavioral_data = pd.read_csv(behavioral_path)
    hcp_data = pd.read_csv(hcp_path)
    behavioral_data = process_age_gender(behavioral_data)
    behavioral_data.to_csv(behavioral_path_processed, index=False)
    hcp_data.to_csv(hcp_path_processed, index=False)


def main():
    try:
        preprocess_hcp()

        behavioral_features = [
            "Subject",
            "Age",
            "Gender",
            "CogFluidComp_Unadj",
            "CogCrystalComp_Unadj",
            "MMSE_Score",
            "NEOFAC_O",
            "NEOFAC_C",
            "NEOFAC_E",
            "NEOFAC_A",
            "NEOFAC_N",
            "ProcSpeed_Unadj",
            "CardSort_Unadj",
            "PicVocab_Unadj",
            "ReadEng_Unadj",
            "PMAT24_A_CR",
            "VSPLOT_TC",
            "IWRD_TOT",
        ]

        hcp_features = [
            "Subject",
            "FS_TotCort_GM_Vol",
            "FS_SubCort_GM_Vol",
            "FS_Total_GM_Vol",
            "FS_Tot_WM_Vol",
            "FS_BrainStem_Vol",
            "FS_L_Hippo_Vol",
            "FS_R_Hippo_Vol",
            "FS_L_Amygdala_Vol",
            "FS_R_Amygdala_Vol",
            "FS_L_Caudate_Vol",
            "FS_R_Caudate_Vol",
            "FS_L_Putamen_Vol",
            "FS_R_Putamen_Vol",
        ]

        categorical_columns = ["Age", "Gender"]

        prior_edges = [
            ("Age", "CogFluidComp_Unadj"),
            ("Age", "CogCrystalComp_Unadj"),
            ("Age", "MMSE_Score"),
            ("Gender", "CogFluidComp_Unadj"),
            ("Gender", "CogCrystalComp_Unadj"),
            ("MMSE_Score", "CogFluidComp_Unadj"),
            ("MMSE_Score", "CogCrystalComp_Unadj"),
            ("FS_Total_GM_Vol", "CogFluidComp_Unadj"),
            ("FS_Total_GM_Vol", "CogCrystalComp_Unadj"),
            ("FS_Tot_WM_Vol", "CogFluidComp_Unadj"),
            ("FS_Tot_WM_Vol", "CogCrystalComp_Unadj"),
            ("FS_L_Hippo_Vol", "CogFluidComp_Unadj"),
            ("FS_R_Hippo_Vol", "CogFluidComp_Unadj"),
            ("FS_L_Amygdala_Vol", "NEOFAC_O"),
            ("FS_R_Amygdala_Vol", "NEOFAC_O"),
            ("NEOFAC_O", "CogCrystalComp_Unadj"),
            ("NEOFAC_C", "CogFluidComp_Unadj"),
            ("FS_L_Hippo_Vol", "NEOFAC_O"),
            ("FS_R_Hippo_Vol", "NEOFAC_O"),
            ("ProcSpeed_Unadj", "CogCrystalComp_Unadj"),
            ("CardSort_Unadj", "CogCrystalComp_Unadj"),
            ("ProcSpeed_Unadj", "MMSE_Score"),
            ("CardSort_Unadj", "MMSE_Score")
        ]

        interaction_features = ["FS_Total_GM_Vol", "FS_Tot_WM_Vol", "Age"]

        target_variable = "CogFluidComp_Unadj"

        params = {
            "target_variable": "CogFluidComp_Unadj",
            "personality_traits": ["NEOFAC_O", "NEOFAC_C"],
            "cognitive_measures": [
                "CogFluidComp_Unadj",
                "MMSE_Score",
                "ProcSpeed_Unadj",
                "CardSort_Unadj",
            ],
            "brain_structure_features": [
                "FS_L_Hippo_Vol",
                "FS_R_Hippo_Vol",
                "FS_L_Amygdala_Vol",
                "FS_R_Amygdala_Vol",
            ],
            "age_column": "Age",
            "gender_column": "Gender",
            "brain_stem_column": "FS_BrainStem_Vol",
            "age_groups": {
                "Young": (0, 1),
                "Adult": (1, 2),
                "Middle": (2, 3),
                "Old": (3, 4),
            },
            "feature_thresholds": {"NEOFAC_O": 0.1, "FS_L_Hippo_Vol": 0.1},
            "analysis_variables": [
                ("FS_L_Amygdala_Vol", "FS_R_Amygdala_Vol"),
                ("FS_L_Hippo_Vol", "FS_R_Hippo_Vol"),
            ],
        }

        data, categorical_columns = prepare_data(
            behavioral_path=behavioral_path,
            hcp_path=hcp_path,
            behavioral_features=behavioral_features,
            hcp_features=hcp_features,
            categorical_columns=categorical_columns,
            index="Subject",
            interaction_features=interaction_features,
        )

        filtered_prior_edges = [(parent, child) for parent, child in prior_edges 
                                if parent != target_variable and child != target_variable]

        data_dict = data.to_dict()

        logger.info("Fitting Bayesian Network")
        
        model_data_future = fit_model.delay(data_dict, categorical_columns, target_variable, filtered_prior_edges)
        model_data = model_data_future.get()

        logger.success("Bayesian Network fitting completed successfully")

        logger.info("Computing edge weights")
        edge_weights_future = compute_edge_weights.delay(model_data)

        logger.info("Explaining network structure")
        structure_explanation_future = explain_structure.delay(model_data)

        logger.info("Getting key relationships")
        key_relationships_future = get_key_relationships.delay(model_data)

        logger.info("Computing sensitivities")
        sensitivities_future = compute_sensitivities.delay(model_data, target_variable)

        logger.info("Performing cross-validation")
        cross_validation_future = cross_validate.delay(model_data, data_dict)

        logger.info("Getting clinical implications")
        clinical_implications_future = get_clinical_implications.delay(model_data)

        logger.info("Analyzing age-dependent relationships")
        age_dependent_relationships_future = analyze_age_dependent_relationships.delay(model_data, data_dict, params["age_column"], target_variable)

        logger.info("Performing interaction effects analysis")
        interaction_effects_future = perform_interaction_effects_analysis.delay(model_data, target_variable)

        logger.info("Performing counterfactual analysis")
        counterfactual_analysis_future = perform_counterfactual_analysis.delay(model_data, data_dict, {params["age_column"]: 30}, target_variable)

        logger.info("Performing sensitivity analysis")
        sensitivity_analysis_future = perform_sensitivity_analysis.delay(model_data, target_variable)

        logger.info("Fitting non-linear model")
        nonlinear_model_data_future = fit_nonlinear_model.delay(model_data, data_dict, target_variable)

        # Retrieve and log results
        edge_weights = edge_weights_future.get()
        for edge, stats in edge_weights.items():
            logger.info(f"Edge {edge}:")
            logger.info(f"  Weight: {stats['weight']:.3f} ± {stats['std_dev']:.3f}")
            logger.info(f"  95% CI: ({stats['95%_CI'][0]:.3f}, {stats['95%_CI'][1]:.3f})")
            logger.info(f"  P(important): {stats['P(important)']:.3f}")

        structure_explanation = structure_explanation_future.get()
        logger.info(f"Structure explanation:\n{structure_explanation}")

        key_relationships = key_relationships_future.get()
        logger.info(f"Key relationships: {key_relationships}")

        sensitivities = sensitivities_future.get()
        logger.info(f"Sensitivities: {sensitivities}")

        mean_ll, std_ll = cross_validation_future.get()
        logger.info(f"Cross-validation results: Mean log-likelihood = {mean_ll:.2f}, Std = {std_ll:.2f}")

        clinical_implications = clinical_implications_future.get()
        logger.info(f"Clinical implications: {clinical_implications}")

        age_dependent_relationships = age_dependent_relationships_future.get()
        logger.info(f"Age-dependent relationships: {age_dependent_relationships}")

        interaction_effects = interaction_effects_future.get()
        logger.info(f"Interaction effects: {interaction_effects}")

        counterfactual_analysis = counterfactual_analysis_future.get()
        logger.info(f"Counterfactual analysis: {counterfactual_analysis}")

        sensitivity_analysis = sensitivity_analysis_future.get()
        logger.info(f"Sensitivity analysis: {sensitivity_analysis}")

        nonlinear_model_data = nonlinear_model_data_future.get()
        logger.info("Non-linear model fitted")

        # Model Comparison
        models = {
            'linear': model_data,
            'nonlinear': nonlinear_model_data
        }

        logger.info("Performing Bayesian model comparison")
        comparison_results_future = bayesian_model_comparison.delay(model_data, data_dict, models)

        logger.info("Analyzing age-related changes")
        age_related_changes_future = analyze_age_related_changes.delay(model_data, data_dict, params["age_column"], params["cognitive_measures"])

        logger.info("Comparing performance to other models")
        performance_comparison_future = compare_performance.delay(model_data, data_dict, target_variable)

        comparison_results = comparison_results_future.get()
        logger.info(f"Model Comparison Results: {comparison_results}")

        age_related_changes = age_related_changes_future.get()
        logger.info(f"Age-related changes: {age_related_changes}")

        performance_comparison = performance_comparison_future.get()
        logger.info(f"Compared performance {performance_comparison}")

        logger.info("Analysis complete.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.exception("Exception details:")

if __name__ == "__main__":
    main()
