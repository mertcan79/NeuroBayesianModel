import logging
import os
import sys
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.data_processing import prepare_data
from src.modeling import BayesianModel
from src.bayesian_network import BayesianNetwork
from utils import write_results_to_json, write_summary_to_json
from archive.other.insights import *

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

environment = os.getenv('ENVIRONMENT', 'local')
data_path = os.getenv('LOCAL_DATA_PATH_PROCESSED') if environment == 'local' else os.getenv('CLOUD_DATA_PATH')

behavioral_path = os.path.join(data_path, 'connectome_behavioral.csv')
hcp_path = os.path.join(data_path, 'hcp_freesurfer.csv')

def main():
    logger.info("Starting Bayesian Network analysis")

    behavioral_features = [
        'Subject', 'Age', 'Gender', 'CogFluidComp_Unadj', 'CogCrystalComp_Unadj', 'MMSE_Score',
        'NEOFAC_O', 'NEOFAC_C', 'ProcSpeed_Unadj', 'CardSort_Unadj', 'PicVocab_Unadj', 'ReadEng_Unadj'
    ]

    hcp_features = [
        'Subject', 'FS_TotCort_GM_Vol', 'FS_SubCort_GM_Vol', 'FS_Total_GM_Vol', 'FS_Tot_WM_Vol', 'FS_BrainStem_Vol',
        'FS_L_Hippo_Vol', 'FS_R_Hippo_Vol', 'FS_L_Amygdala_Vol', 'FS_R_Amygdala_Vol',
        'FS_L_Caudate_Vol', 'FS_R_Caudate_Vol', 'FS_L_Putamen_Vol', 'FS_R_Putamen_Vol',
    ]

    categorical_columns = ['Age', 'Gender']

    prior_edges = [
        ('Age', 'CogFluidComp_Unadj'), ('Age', 'CogCrystalComp_Unadj'), ('Age', 'MMSE_Score'),
        ('Gender', 'CogFluidComp_Unadj'), ('Gender', 'CogCrystalComp_Unadj'),
        ('MMSE_Score', 'CogFluidComp_Unadj'), ('MMSE_Score', 'CogCrystalComp_Unadj'),
        ('FS_Total_GM_Vol', 'CogFluidComp_Unadj'), ('FS_Total_GM_Vol', 'CogCrystalComp_Unadj'),
        ('FS_Tot_WM_Vol', 'CogFluidComp_Unadj'), ('FS_Tot_WM_Vol', 'CogCrystalComp_Unadj'),
        ('FS_L_Hippo_Vol', 'CogFluidComp_Unadj'), ('FS_R_Hippo_Vol', 'CogFluidComp_Unadj'),
        ('FS_L_Amygdala_Vol', 'NEOFAC_O'), ('FS_R_Amygdala_Vol', 'NEOFAC_O'),
        ('NEOFAC_O', 'CogCrystalComp_Unadj'), ('NEOFAC_C', 'CogFluidComp_Unadj'),
        ('FS_L_Hippo_Vol', 'NEOFAC_O'), ('FS_R_Hippo_Vol', 'NEOFAC_O'),
        ('ProcSpeed_Unadj', 'CogFluidComp_Unadj'), ('CardSort_Unadj', 'CogFluidComp_Unadj'),
        ('ProcSpeed_Unadj', 'CogCrystalComp_Unadj'), ('CardSort_Unadj', 'CogCrystalComp_Unadj'),
        ('ProcSpeed_Unadj', 'MMSE_Score'), ('CardSort_Unadj', 'MMSE_Score'),
        ('ProcSpeed_Unadj', 'NEOFAC_O'), ('CardSort_Unadj', 'NEOFAC_C'),
    ]

    features_to_remove = ['FS_Total_GM_Vol', 'FS_TotCort_GM_Vol', 'CogCrystalComp_Unadj']

    behavioral_features = [f for f in behavioral_features if f not in features_to_remove]
    hcp_features = [f for f in hcp_features if f not in features_to_remove]
    prior_edges = [edge for edge in prior_edges if edge[0] not in features_to_remove and edge[1] not in features_to_remove]

    logger.info("Preparing data...")
    data, categorical_columns, categories = prepare_data(
        behavioral_path=behavioral_path,
        hcp_path=hcp_path,
        behavioral_features=behavioral_features,
        hcp_features=hcp_features,
        categorical_columns=categorical_columns
    )

    # Perform data analysis
    results = {}

    # Create and analyze Bayesian Network
    logger.info("Creating Bayesian Network")
    model = BayesianModel(method='nsl', max_parents=6, iterations=1500, categorical_columns=categorical_columns)
    model.fit(data, prior_edges=prior_edges)

    required_nodes = ["CogFluidComp_Unadj", "CogCrystalComp_Unadj", "MMSE_Score", "NEOFAC_O", "NEOFAC_C", "ProcSpeed_Unadj", "CardSort_Unadj"]
    missing_nodes = [node for node in required_nodes if node not in model.network.nodes]
    if missing_nodes:
        logger.error(f"Missing nodes in the Bayesian Network: {', '.join(missing_nodes)}")
        return

    logger.info("Analyzing Bayesian Network")
    results['network_structure'] = model.network.explain_structure_extended()
    results['edge_probabilities'] = model.network.compute_edge_probabilities()
    results['key_relationships'] = model.network.get_key_relationships()
    results['feature_importance'] = model.network.compute_feature_importance(model, data)

    # Simulate interventions
    logger.info("Simulating interventions")
    interventions = {
        'FS_Tot_WM_Vol': lambda x: x * 1.1,  # Increase white matter volume by 10%
        'NEOFAC_O': lambda x: min(x + 1, 5),  # Increase openness score by 1 (max 5)
        'ProcSpeed_Unadj': lambda x: x * 1.2,  # Increase processing speed by 20%
        'FS_L_Hippo_Vol': lambda x: x * 1.05,  # Increase left hippocampus volume by 5%
    }
    simulated_data = model.simulate_intervention(interventions)
    
    results["simulated_data_summary"] = simulated_data.describe().to_dict()

    # Compute sensitivity
    logger.info("Computing sensitivity")
    results['sensitivity'] = model.network.compute_sensitivity(model, data)

    # Write results to JSON
    logger.info("Writing results and summary to JSON")
    write_results_to_json(model.network, data, results)
    write_summary_to_json(model.network, results)

    logger.info("Analysis complete.")