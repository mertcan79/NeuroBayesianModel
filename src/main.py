import logging
from data_processing import prepare_data
from modeling import create_bayesian_network, analyze_network, create_hierarchical_bayesian_network
from bayesian_network import BayesianNetwork
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Bayesian Network analysis")

    # Prepare data
    data, categorical_columns = prepare_data()

    # Define Bayesian Network structure
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
        ('FS_TotCort_GM_Vol', 'CogCrystalComp_Unadj'),
        ('FS_BrainStem_Vol', 'ProcSpeed_Unadj'),
        ('FS_L_Putamen_Vol', 'CardSort_Unadj'),
        ('FS_R_Putamen_Vol', 'CardSort_Unadj'),
        ('FS_L_Hippo_Vol', 'NEOFAC_O'),
        ('FS_R_Hippo_Vol', 'NEOFAC_O'),
    ]

    # Filter prior_edges to only include edges between existing columns
    existing_columns = set(data.columns)
    filtered_prior_edges = [
        (parent, child) for parent, child in prior_edges
        if parent in existing_columns and child in existing_columns
    ]

    # Create and analyze Bayesian Network
    logger.info("Creating Bayesian Network")
    model = create_bayesian_network(data, categorical_columns, filtered_prior_edges)

    logger.info("Analyzing Bayesian Network")
    results = analyze_network(model, data)

    # Write results to JSON
    logger.info("Writing results to JSON")
    model.write_results_to_json(results)

    # Create Hierarchical Bayesian Network
    logger.info("Creating Hierarchical Bayesian Network")
    hierarchical_levels = ['cellular', 'regional', 'functional']
    level_constraints = {
        "functional": ["regional"],
        "regional": ["cellular"]
    }
    h_model = create_hierarchical_bayesian_network(data, categorical_columns, hierarchical_levels, level_constraints)

    logger.info("Hierarchical Network Structure:")
    logger.info(h_model.explain_structure_extended())

    logger.info("Analysis complete.")
