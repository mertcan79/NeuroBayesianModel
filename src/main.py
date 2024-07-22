import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from bayesian_network import BayesianNetwork, HierarchicalBayesianNetwork
from typing import List
import logging
from tasks import fit_bayesian_network, compute_sensitivity, cross_validate, fit_hierarchical_bayesian_network
from celery.exceptions import TimeoutError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(data: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    data = data.copy()
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in categorical_columns]

    numeric_imputer = SimpleImputer(strategy='median')
    data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])

    categorical_imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])

    for col in categorical_columns:
        data[col] = pd.Categorical(data[col]).codes

    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    return data

def run_celery_task(task, *args, **kwargs):
    try:
        result = task.delay(*args, **kwargs)
        return result.get(timeout=300)  # 5 minutes timeout
    except TimeoutError:
        logger.error(f"Task {task.__name__} timed out")
        return None
    except Exception as e:
        logger.error(f"Error running task {task.__name__}: {str(e)}")
        return None

def main():
    logger.info("Starting Bayesian Network analysis")

    # Load data
    behavioral = pd.read_csv('/Users/macbookair/Documents/bayesian_dataclass/data/connectome_behavioral.csv')
    hcp = pd.read_csv('/Users/macbookair/Documents/bayesian_dataclass/data/hcp_freesurfer.csv')

    # Select relevant features
    relevant_features_behavioral = [
        'Subject', 'Age', 'Gender', 'CogFluidComp_Unadj', 'CogCrystalComp_Unadj', 'MMSE_Score',
        'NEOFAC_O', 'NEOFAC_C', 'ProcSpeed_Unadj', 'CardSort_Unadj', 'PicVocab_Unadj', 'ReadEng_Unadj'
    ]

    relevant_features_hcp = [
        'Subject', 'Gender', 'FS_TotCort_GM_Vol', 'FS_SubCort_GM_Vol', 'FS_Total_GM_Vol', 'FS_Tot_WM_Vol', 'FS_BrainStem_Vol',
        'FS_L_Hippo_Vol', 'FS_R_Hippo_Vol', 'FS_L_Amygdala_Vol', 'FS_R_Amygdala_Vol',
        'FS_L_Caudate_Vol', 'FS_R_Caudate_Vol', 'FS_L_Putamen_Vol', 'FS_R_Putamen_Vol',
    ]

    relevant_features_hcp_temporal = [col for col in hcp.columns if 'temporal' in col.lower()]
    relevant_features_hcp = list(set(relevant_features_hcp + relevant_features_hcp_temporal))

    hcp = hcp[relevant_features_hcp].copy()
    behavioral = behavioral[relevant_features_behavioral].copy()

    # Merge the datasets
    data = pd.merge(hcp, behavioral, on=["Subject", "Gender"])

    # Specify categorical columns
    categorical_columns = ['Gender', 'Age', 'MMSE_Score']

    # Preprocess data
    df_processed = preprocess_data(data, categorical_columns)

    # Define Bayesian Network structure with more specific connections
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
    existing_columns = set(df_processed.columns)
    filtered_prior_edges = [
        (parent, child) for parent, child in prior_edges
        if parent in existing_columns and child in existing_columns
    ]

    logger.info("Fitting Bayesian Network")
    model_dict = run_celery_task(fit_bayesian_network, df_processed.to_dict(), filtered_prior_edges, categorical_columns)
    if model_dict is None:
        logger.error("Failed to fit Bayesian Network. Exiting.")
        return

    try:
        model = BayesianNetwork.from_dict(model_dict)
    except Exception as e:
        logger.error(f"Error creating BayesianNetwork from dict: {str(e)}")
        return

    model = BayesianNetwork.from_dict(model_dict)

    logger.info("Computing log-likelihood")
    log_likelihood = model.log_likelihood(df_processed)
    logger.info(f"Log-likelihood: {log_likelihood}")

    logger.info("Performing cross-validation")
    cv_result = run_celery_task(cross_validate, model_dict, df_processed.to_dict(), k_folds=5)
    if cv_result is not None:
        mean_ll, std_ll = cv_result
        logger.info(f"Cross-validation: Mean LL = {mean_ll:.4f}, Std = {std_ll:.4f}")
    else:
        logger.error("Cross-validation failed")

    logger.info("Computing sensitivity")
    sensitivity = run_celery_task(compute_sensitivity, model_dict, 'CogFluidComp_Unadj', num_samples=1000)
    
    if isinstance(sensitivity, dict):
        top_sensitivities = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("\nTop 10 sensitivities for CogFluidComp_Unadj:")
        for node, value in top_sensitivities:
            logger.info(f"Sensitivity to {node}: {value:.4f}")
    else:
        logger.error("Sensitivity computation failed or did not return a dictionary.")

    logger.info("Explaining network structure")
    network_structure = model.explain_structure_extended()

    logger.info("Performing Metropolis-Hastings sampling")
    observed_data = {
        'Age': 0,
        'Gender': 1,
        'MMSE_Score': 0,
    }
    mh_samples = model.metropolis_hastings(observed_data, num_samples=1000)

    logger.info("Metropolis-Hastings sampling results:")
    for node, samples in mh_samples.items():
        if node not in observed_data:
            logger.info(f"{node}: Mean = {np.mean(samples):.4f}, Std = {np.std(samples):.4f}")

    # Prepare results
    results = {
        "log_likelihood": log_likelihood,
        "cross_validation": {"mean": mean_ll, "std": std_ll} if 'mean_ll' in locals() else None,
        "top_sensitivities": dict(top_sensitivities) if isinstance(sensitivity, dict) else {},
        "network_structure": network_structure,
        "mh_samples": {node: {"mean": float(np.mean(samples)), "std": float(np.std(samples))} 
                       for node, samples in mh_samples.items() if node not in observed_data}
    }

    # Write results to JSON
    model.write_results_to_json(results)

    logger.info("Fitting Hierarchical Bayesian Network")
    hierarchical_levels = ['cellular', 'regional', 'functional']
    level_constraints = {
        "functional": ["regional"],
        "regional": ["cellular"]
    }
    h_model_dict = run_celery_task(fit_hierarchical_bayesian_network,
        df_processed.to_dict(), hierarchical_levels, level_constraints, categorical_columns
    )
    if h_model_dict is not None:
        h_model = HierarchicalBayesianNetwork.from_dict(h_model_dict)
        logger.info("Hierarchical Network Structure:")
        logger.info(h_model.explain_structure_extended())
    else:
        logger.error("Failed to fit Hierarchical Bayesian Network")

    logger.info("Analysis complete.")

if __name__ == "__main__":
    main()