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

    # Select a smaller subset of columns to reduce complexity
    selected_columns = [
        'Age', 'Gender', 'MMSE_Score', 'CogFluidComp_Unadj', 'CogCrystalComp_Unadj',
        'FS_Total_GM_Vol'
    ]
    data = data[selected_columns]
    categorical_columns = [col for col in categorical_columns if col in selected_columns]

    # Define a simpler Bayesian Network structure
    prior_edges: List[Tuple[str, str]] = [
        ('Age', 'CogFluidComp_Unadj'),
        ('Age', 'CogCrystalComp_Unadj'),
        ('Gender', 'CogFluidComp_Unadj'),
        ('MMSE_Score', 'CogFluidComp_Unadj'),
        ('FS_Total_GM_Vol', 'CogFluidComp_Unadj'),
    ]

    # Create and analyze Bayesian Network
    logger.info("Creating Bayesian Network")
    model = create_bayesian_network(data, categorical_columns, prior_edges)

    logger.info("Analyzing Bayesian Network")
    results = analyze_network(model, data)

    # Write results to JSON
    logger.info("Writing results to JSON")
    model.write_results_to_json(results)

    # Create Hierarchical Bayesian Network
    logger.info("Creating Hierarchical Bayesian Network")
    hierarchical_levels = ['cellular', 'functional']
    level_constraints = {
        "functional": ["cellular"]
    }
    h_model = create_hierarchical_bayesian_network(data, categorical_columns, hierarchical_levels, level_constraints)

    logger.info("Hierarchical Network Structure:")
    logger.info(h_model.explain_structure_extended())

    logger.info("Analysis complete.")

if __name__ == "__main__":
    main()