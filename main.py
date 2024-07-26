import logging
from logging_config import setup_logging
import os
from dotenv import load_dotenv
import pandas as pd
from src.data_processing import prepare_data
from src.modeling import BayesianModel
from src.bayesian_network import BayesianNetwork
from utils import write_results_to_json, write_summary_to_json
import numpy as np

# Load environment variables
load_dotenv()

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Define environment and file paths
environment = os.getenv('ENVIRONMENT', 'local')
data_path = os.getenv('LOCAL_DATA_PATH') if environment == 'local' else os.getenv('CLOUD_DATA_PATH')
processed_data_path = os.getenv('LOCAL_DATA_PATH_PROCESSED') if environment == 'local' else os.getenv('CLOUD_DATA_PATH')

# File paths
behavioral_path = os.path.join(data_path, 'connectome_behavioral.csv')
behavioral_path_processed = os.path.join(processed_data_path, 'connectome_behavioral.csv')
hcp_path = os.path.join(data_path, 'hcp_freesurfer.csv')
hcp_path_processed = os.path.join(processed_data_path, 'hcp_freesurfer.csv')

def preprocess_hcp():
    def map_age_to_category(age_str):
        bins = ['22-25', '26-30', '31-35', '36+']
        categories = [1, 2, 3, 4]
        
        if pd.isna(age_str):
            return np.nan
        age_str = age_str.strip()
        
        if age_str in bins:
            return categories[bins.index(age_str)]
        else:
            return np.nan

    def process_age_gender(data: pd.DataFrame) -> pd.DataFrame:
        if 'Age' in data.columns and data['Age'].dtype == 'object':
            data['Age'] = data['Age'].apply(map_age_to_category)
        return data

    try:
        # Load your data
        behavioral_data = pd.read_csv(behavioral_path)
        hcp_data = pd.read_csv(hcp_path)
        
        # Process Age column
        behavioral_data = process_age_gender(behavioral_data)

        # Save the processed data
        behavioral_data.to_csv(behavioral_path_processed, index=False)
        hcp_data.to_csv(hcp_path_processed, index=False)
        logger.info("Preprocessing completed and data saved.")
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")

def main():
    try:
        preprocess_hcp()
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

        logger.info("Creating Bayesian Network")
        model = BayesianModel(method='nsl', max_parents=6, iterations=1500, categorical_columns=categorical_columns)
        model.fit(data, prior_edges=prior_edges)

        required_nodes = ["CogFluidComp_Unadj", "MMSE_Score", "NEOFAC_O", "NEOFAC_C", "ProcSpeed_Unadj", "CardSort_Unadj"]
        missing_nodes = [node for node in required_nodes if node not in model.network.nodes]
        if missing_nodes:
            logger.error(f"Missing nodes in the Bayesian Network: {', '.join(missing_nodes)}")
            return

        logger.info("Analyzing Bayesian Network and writing results")
        results = {}
        write_results_to_json(model.network, data, results)
        write_summary_to_json(model.network, results)

        logger.info("Analysis complete.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
