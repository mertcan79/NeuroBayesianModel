import logging
import os
import sys
from dotenv import load_dotenv
import pandas as pd
from src.data_processing import prepare_data
from src.modeling import BayesianModel
from src.bayesian_network import BayesianNetwork

load_dotenv()

# Determine environment and data path
environment = os.getenv('ENVIRONMENT', 'local')
data_path = os.getenv('LOCAL_DATA_PATH') if environment == 'local' else os.getenv('CLOUD_DATA_PATH')

# File paths
behavioral_path = os.path.join(data_path, 'connectome_behavioral.csv')
hcp_path = os.path.join(data_path, 'hcp_freesurfer.csv')

# Feature selections
behavioral_features = [
    'Subject', 'Age', 'Gender', 'CogFluidComp_Unadj', 'CogCrystalComp_Unadj', 'MMSE_Score',
    'NEOFAC_O', 'NEOFAC_C', 'ProcSpeed_Unadj', 'CardSort_Unadj', 'PicVocab_Unadj', 'ReadEng_Unadj'
]
hcp_features = [
    'Subject', 'FS_TotCort_GM_Vol', 'FS_SubCort_GM_Vol', 'FS_Total_GM_Vol', 'FS_Tot_WM_Vol', 'FS_BrainStem_Vol',
    'FS_L_Hippo_Vol', 'FS_R_Hippo_Vol', 'FS_L_Amygdala_Vol', 'FS_R_Amygdala_Vol',
    'FS_L_Caudate_Vol', 'FS_R_Caudate_Vol', 'FS_L_Putamen_Vol', 'FS_R_Putamen_Vol',
]
categorical_columns_hcp = ['Gender', 'MMSE_Score', 'Age']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Bayesian Network analysis")

    # Prepare data
    logger.info("Preparing data...")
    data, categorical_columns, categories = prepare_data(
        behavioral_path=behavioral_path,
        hcp_path=hcp_path,
        behavioral_features=behavioral_features,
        hcp_features=hcp_features,
        categorical_columns=categorical_columns_hcp
    )

    # Sample data
    data = data.sample(n=200, random_state=42)

    # Define prior edges
    prior_edges = [
        ('Age', 'CogFluidComp_Unadj'),
        ('Age', 'CogCrystalComp_Unadj'),
        ('Gender', 'CogFluidComp_Unadj'),
        ('Gender', 'CogCrystalComp_Unadj'),
        ('FS_Total_GM_Vol', 'CogFluidComp_Unadj'),
        ('FS_Total_GM_Vol', 'CogCrystalComp_Unadj'),
        ('FS_Tot_WM_Vol', 'CogFluidComp_Unadj'),
        ('FS_Tot_WM_Vol', 'CogCrystalComp_Unadj'),
        ('NEOFAC_O', 'CogCrystalComp_Unadj'),
        ('NEOFAC_C', 'CogFluidComp_Unadj'),
    ]

    # Create and analyze Bayesian Network
    logger.info("Creating Bayesian Network")
    model = BayesianModel(method='k2', max_parents=2, iterations=100, categorical_columns=categorical_columns)
    model.fit(data, prior_edges=prior_edges)

    logger.info("Analyzing Bayesian Network")
    results = {}
    results['network_structure'] = model.network.explain_structure_extended()
    results['edge_probabilities'] = model.network.compute_edge_probabilities()
    results['key_relationships'] = model.network.get_key_relationships()

    # Write results to JSON
    logger.info("Writing results to JSON")
    model.write_results_to_json(results)

    logger.info("Network Structure:")
    logger.info(model.network.explain_structure_extended())

    # Simulate intervention
    logger.info("Simulating intervention")
    interventions = {
        'FS_Total_GM_Vol': lambda x: x * 1.1,  # Increase gray matter volume by 10%
        'NEOFAC_O': lambda x: min(x + 1, 5)    # Increase openness score by 1 (max 5)
    }
    simulated_data = model.simulate_intervention(interventions)
    
    logger.info(f"Simulated data shape: {simulated_data.shape}")
    logger.info("Summary of simulated data:")
    logger.info(simulated_data.describe())

    # Add simulated data summary to results
    results["simulated_data_summary"] = simulated_data.describe().to_dict()
    model.write_summary_to_json(results)

    logger.info("Analysis complete.")

if __name__ == "__main__":
    main()