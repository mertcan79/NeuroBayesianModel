import logging
import os
import sys
from dotenv import load_dotenv
import pandas as pd
from src.data_processing import prepare_data
from src.modeling import BayesianModel
from src.bayesian_network import HierarchicalBayesianNetwork

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

# Add 'src' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

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

    # Define prior edges
    prior_edges = [
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
        ('NEOFAC_O', 'CogCrystalComp_Unadj'),
        ('NEOFAC_C', 'CogFluidComp_Unadj'),
        ('FS_L_Hippo_Vol', 'NEOFAC_O'),
        ('FS_R_Hippo_Vol', 'NEOFAC_O'),
    ]

    # Create and analyze Bayesian Network
    logger.info("Creating Bayesian Network")
    model = BayesianModel(method='hill_climb', max_parents=2, iterations=300, categorical_columns=categorical_columns)
    model.fit(data, prior_edges=prior_edges)

    logger.info("Analyzing Bayesian Network")
    results = analyze_network(model, data)

    # Write results to JSON
    logger.info("Writing results to JSON")
    model.write_results_to_json(results)
    model.write_summary_to_json(results)

    logger.info("Network Structure:")
    logger.info(model.explain_structure_extended())

    # Simulate intervention
    logger.info("Simulating intervention")
    interventions = {
        'Age': categories['Age'][2],  # Example: Select a specific category
        'Gender': categories['Gender'][1],  # Example: Select a specific category
        'FS_Total_GM_Vol': data['FS_Total_GM_Vol'].mean() + data['FS_Total_GM_Vol'].std(),
        'FS_Tot_WM_Vol': data['FS_Tot_WM_Vol'].mean() + data['FS_Tot_WM_Vol'].std(),
    }
    simulated_data = model.simulate_intervention(interventions)
    
    logger.info(f"Simulated data shape: {simulated_data.shape}")
    logger.info("Summary of simulated data:")
    logger.info(simulated_data.describe())

    # Add simulated data summary to results
    results["simulated_data_summary"] = simulated_data.describe().to_dict()
    model.write_summary_to_json(results)

    # Hierarchical Bayesian Network
    logger.info("Creating Hierarchical Bayesian Network")
    levels = ['Demographics', 'Brain_Structure', 'Cognitive_Function', 'Personality']
    level_constraints = {
        'Demographics': ['Age', 'Gender'],
        'Brain_Structure': ['FS_TotCort_GM_Vol', 'FS_SubCort_GM_Vol', 'FS_Total_GM_Vol', 'FS_Tot_WM_Vol', 'FS_BrainStem_Vol',
                            'FS_L_Hippo_Vol', 'FS_R_Hippo_Vol', 'FS_L_Amygdala_Vol', 'FS_R_Amygdala_Vol',
                            'FS_L_Caudate_Vol', 'FS_R_Caudate_Vol', 'FS_L_Putamen_Vol', 'FS_R_Putamen_Vol'],
        'Cognitive_Function': ['CogFluidComp_Unadj', 'CogCrystalComp_Unadj', 'MMSE_Score', 'ProcSpeed_Unadj', 'CardSort_Unadj', 'PicVocab_Unadj', 'ReadEng_Unadj'],
        'Personality': ['NEOFAC_O', 'NEOFAC_C']
    }
    h_model = HierarchicalBayesianNetwork(levels=levels, method='hill_climb', max_parents=2, iterations=300, categorical_columns=categorical_columns)
    h_model.fit(data, level_constraints=level_constraints)

    logger.info("Analyzing Hierarchical Bayesian Network")
    h_results = analyze_network(h_model, data)
    h_model.write_results_to_json(h_results)
    h_model.write_summary_to_json(h_results)

    #logger.info("Hierarchical Network Structure:")
    #logger.info(h_model.explain_hierarchical_structure())

    logger.info("Analysis complete.")

if __name__ == "__main__":
    main()
