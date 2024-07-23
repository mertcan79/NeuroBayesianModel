import logging
import pandas as pd
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
    target_node = "CogFluidComp_Unadj"  # Example target node
    sensitivity = model.compute_sensitivity(target_node)
    results["sensitivity"] = sensitivity
    
    # You can add more analyses here
    
    return results

def main():
    logger.info("Starting Bayesian Network analysis")

    # Prepare data
    data, categorical_columns = prepare_data()

    # Select a smaller subset of columns to reduce complexity
    selected_columns = ['Age', 'Gender', 'MMSE_Score', 'CogFluidComp_Unadj']
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
    intervention = {'Age': 65}  # Example intervention
    simulated_data = model.simulate_intervention(intervention)
    logger.info(f"Simulated data shape: {simulated_data.shape}")

    logger.info("Analysis complete.")

if __name__ == "__main__":
    main()