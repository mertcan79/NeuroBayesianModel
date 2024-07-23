import logging
import pandas as pd
import numpy as np
from typing import List, Tuple
from src.data_processing import prepare_data
from src.modeling import BayesianModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_network(model: BayesianModel, data: pd.DataFrame) -> dict:
    results = {}
    
    # Evaluate the model
    logger.info("Evaluating the model")
    mean_log_likelihood, std_log_likelihood = model.evaluate(data)
    results["mean_log_likelihood"] = mean_log_likelihood
    results["std_log_likelihood"] = std_log_likelihood
    
    # Compute sensitivity
    logger.info("Computing sensitivity")
    target_nodes = ["CogFluidComp_Unadj", "CogCrystalComp_Unadj"]
    sensitivity = {}
    for target_node in target_nodes:
        sensitivity[target_node] = model.compute_sensitivity(target_node)
    results["sensitivity"] = sensitivity
    
    return results

def main():
    logger.info("Starting Bayesian Network analysis")

    # Prepare data
    data, categorical_columns = prepare_data()

    # Select a larger subset of columns
    selected_columns = [
        'Subject', 'Age', 'Gender', 'CogFluidComp_Unadj', 'CogCrystalComp_Unadj', 'MMSE_Score',
        'NEOFAC_O', 'NEOFAC_C', 'ProcSpeed_Unadj', 'CardSort_Unadj', 'PicVocab_Unadj', 'ReadEng_Unadj',
        'FS_TotCort_GM_Vol', 'FS_SubCort_GM_Vol', 'FS_Total_GM_Vol', 'FS_Tot_WM_Vol', 'FS_BrainStem_Vol',
        'FS_L_Hippo_Vol', 'FS_R_Hippo_Vol', 'FS_L_Amygdala_Vol', 'FS_R_Amygdala_Vol',
        'FS_L_Caudate_Vol', 'FS_R_Caudate_Vol', 'FS_L_Putamen_Vol', 'FS_R_Putamen_Vol'
    ]
    data = data[selected_columns]
    categorical_columns = [col for col in categorical_columns if col in selected_columns]

    # Define a more complex Bayesian Network structure
    prior_edges: List[Tuple[str, str]] = [
        ('Age', 'CogFluidComp_Unadj'),
        ('Age', 'CogCrystalComp_Unadj'),
        ('Age', 'MMSE_Score'),
        ('Gender', 'CogFluidComp_Unadj'),
        ('Gender', 'CogCrystalComp_Unadj'),
        ('MMSE_Score', 'CogFluidComp_Unadj'),
        ('MMSE_Score', 'CogCrystalComp_Unadj'),
        ('FS_Total_GM_Vol', 'CogFluidComp_Unadj'),
        ('FS_Total_GM_Vol', 'CogCrystalComp_Unadj'),
        ('FS_Tot_WM_Vol', 'CogFluidComp_Unadj'),
        ('FS_Tot_WM_Vol', 'CogCrystalComp_Unadj'),
        ('FS_L_Hippo_Vol', 'CogFluidComp_Unadj'),
        ('FS_R_Hippo_Vol', 'CogFluidComp_Unadj'),
        ('FS_L_Amygdala_Vol', 'NEOFAC_O'),
        ('FS_R_Amygdala_Vol', 'NEOFAC_O'),
        ('FS_L_Caudate_Vol', 'ProcSpeed_Unadj'),
        ('FS_R_Caudate_Vol', 'ProcSpeed_Unadj'),
        ('NEOFAC_O', 'CogCrystalComp_Unadj'),
        ('NEOFAC_C', 'CogFluidComp_Unadj'),
        ('CogFluidComp_Unadj', 'ProcSpeed_Unadj'),
        ('CogFluidComp_Unadj', 'CardSort_Unadj'),
        ('CogCrystalComp_Unadj', 'PicVocab_Unadj'),
        ('CogCrystalComp_Unadj', 'ReadEng_Unadj'),
    ]

    # Create and analyze Bayesian Network
    logger.info("Creating Bayesian Network")
    model = BayesianModel(categorical_columns=categorical_columns)
    model.fit(data, prior_edges=prior_edges)

    logger.info("Analyzing Bayesian Network")
    results = analyze_network(model, data)

    # Write results to JSON
    logger.info("Writing results to JSON")
    model.write_results_to_json(results)

    logger.info("Network Structure:")
    logger.info(model.explain_structure_extended())

    # Simulate intervention
    logger.info("Simulating intervention")
    interventions = {
        'Age': 65,
        'Gender': 1,  # Assuming 1 represents one gender category
        'FS_Total_GM_Vol': data['FS_Total_GM_Vol'].mean() + data['FS_Total_GM_Vol'].std(),  # Increase by 1 std dev
        'FS_Tot_WM_Vol': data['FS_Tot_WM_Vol'].mean() + data['FS_Tot_WM_Vol'].std(),  # Increase by 1 std dev
    }
    simulated_data = model.simulate_intervention(interventions)
    
    logger.info(f"Simulated data shape: {simulated_data.shape}")
    logger.info("Summary of simulated data:")
    logger.info(simulated_data.describe())

    # Add simulated data to results
    results["simulated_data"] = simulated_data.to_dict()
    model.write_results_to_json(results)

    logger.info("Analysis complete.")

if __name__ == "__main__":
    main()