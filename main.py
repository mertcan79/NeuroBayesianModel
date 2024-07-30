import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
from logging_config import setup_logging  # Assuming logging_config.py is in the root
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from src.data_processing import prepare_data
from src.modeling import BayesianModel
from src.utils import calculate_results, write_results_to_json


load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)

environment = os.getenv('ENVIRONMENT', 'local')
data_path = os.getenv('LOCAL_DATA_PATH') if environment == 'local' else os.getenv('CLOUD_DATA_PATH')
processed_data_path = os.getenv('LOCAL_DATA_PATH_PROCESSED') if environment == 'local' else os.getenv('CLOUD_DATA_PATH')

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

    behavioral_data = pd.read_csv(behavioral_path)
    hcp_data = pd.read_csv(hcp_path)
    behavioral_data = process_age_gender(behavioral_data)
    behavioral_data.to_csv(behavioral_path_processed, index=False)
    hcp_data.to_csv(hcp_path_processed, index=False)

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
            ('ProcSpeed_Unadj', 'CogFluidComp_Unadj'),
            ('CardSort_Unadj', 'CogFluidComp_Unadj'),
            ('ProcSpeed_Unadj', 'CogCrystalComp_Unadj'),
            ('CardSort_Unadj', 'CogCrystalComp_Unadj'),
            ('ProcSpeed_Unadj', 'MMSE_Score'),
            ('CardSort_Unadj', 'MMSE_Score'),
            ('ProcSpeed_Unadj', 'NEOFAC_O'),
            ('CardSort_Unadj', 'NEOFAC_C'),
        ]

        data, categorical_columns, categories = prepare_data(
            behavioral_path=behavioral_path,
            hcp_path=hcp_path,
            behavioral_features=behavioral_features,
            hcp_features=hcp_features,
            categorical_columns=categorical_columns,
            index='Subject'
        )

        model = BayesianModel(method='nsl', max_parents=6, iterations=1500, categorical_columns=categorical_columns)
        model.fit(data, prior_edges=prior_edges)

        mean_ll, std_ll = model.cross_validate(data)
        print(f"Cross-validated Log-Likelihood: {mean_ll:.2f} (+/- {std_ll:.2f})")

        total_ll = sum(model.compute_log_likelihood(sample) for _, sample in data.iterrows())
        print(f"Total Log-Likelihood: {total_ll:.2f}")

        analysis_params = {
            "target_variable": "CogFluidComp_Unadj",
            "personality_traits": ["NEOFAC_O", "NEOFAC_C"],
            "cognitive_measures": ["CogFluidComp_Unadj", "MMSE_Score", "ProcSpeed_Unadj", "CardSort_Unadj"],
            "brain_structure_features": ["FS_L_Hippo_Vol", "FS_R_Hippo_Vol", "FS_L_Amygdala_Vol", "FS_R_Amygdala_Vol"],
            "age_column": "Age",
            "gender_column": "Gender",
            "brain_stem_column": "FS_BrainStem_Vol",
            "age_groups": {"Young": (0, 1), "Adult": (1, 2), "Middle": (2, 3), "Old": (3, 4)}, 
            "feature_thresholds": {"NEOFAC_O": 0.1, "FS_L_Hippo_Vol": 0.1},
            "analysis_variables": [('FS_L_Amygdala_Vol', 'FS_R_Amygdala_Vol'), ('FS_L_Hippo_Vol', 'FS_R_Hippo_Vol')]
        }

        try:
            results = calculate_results(model.network, data, analysis_params)
            write_results_to_json(results)
            logger.info("Analysis complete.")
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            logger.exception("Exception details:")

        logger.info("Analysis complete.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.exception("Exception details:")

if __name__ == "__main__":
    main()