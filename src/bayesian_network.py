import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesianNetwork:
    def __init__(self, max_parents=3):
        self.max_parents = max_parents
        self.nodes: Dict[str, Any] = {}
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def fit(self, data: pd.DataFrame, prior_edges: List[tuple] = None, progress_callback: Callable[[float], None] = None):
        try:
            self._learn_structure(data, prior_edges)
            if progress_callback:
                progress_callback(0.5)
            self._fit_parameters(data)
            if progress_callback:
                progress_callback(1.0)
        except Exception as e:
            logger.error(f"Error during fitting: {str(e)}")
            raise

    def _learn_structure(self, data: pd.DataFrame, prior_edges: List[tuple] = None):
        imputed_data = self.imputer.fit_transform(data)
        scaled_data = self.scaler.fit_transform(imputed_data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

        hc = HillClimbSearch(data=scaled_data)
        est_model = hc.estimate(scoring_method=BicScore(data=scaled_data), max_indegree=self.max_parents)

        self.model = BayesianModel(est_model.edges())

        if prior_edges:
            for edge in prior_edges:
                if edge not in self.model.edges():
                    self.model.add_edge(edge[0], edge[1])

        self.nodes = {node: {'parents': self.model.get_parents(node), 'children': self.model.get_children(node)} 
                      for node in self.model.nodes()}

        logger.info("Learned graph structure:")
        for node, data in self.nodes.items():
            logger.info(f"{node} -> parents: {data['parents']}, children: {data['children']}")

    def _fit_parameters(self, data: pd.DataFrame):
        imputed_data = self.imputer.transform(data)
        scaled_data = self.scaler.transform(imputed_data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

        for node_name, node in self.nodes.items():
            node_data = scaled_data[node_name].values
            if not node['parents']:
                mean, std = np.mean(node_data), np.std(node_data)
                node['distribution'] = stats.norm(loc=mean, scale=std)
            else:
                parent_data = scaled_data[node['parents']].values
                beta = np.linalg.lstsq(parent_data, node_data, rcond=None)[0]
                residuals = node_data - parent_data.dot(beta)
                std = np.std(residuals)
                node['distribution'] = lambda x, beta=beta, std=std: stats.norm(
                    loc=x.dot(beta), scale=std
                )

        logger.info("Fitted parameters")

    def sample_node(self, node_name: str, size: int = 1) -> np.ndarray:
        node = self.nodes[node_name]
        if not node['parents']:
            samples = node['distribution'].rvs(size=size)
        else:
            parent_values = np.array([self.sample_node(parent, size) for parent in node['parents']]).T
            samples = node['distribution'](parent_values).rvs(size=size)
        return self.scaler.inverse_transform(samples.reshape(-1, 1)).flatten()

    def log_likelihood(self, data: pd.DataFrame) -> float:
        imputed_data = self.imputer.transform(data)
        scaled_data = self.scaler.transform(imputed_data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

        log_likelihood = 0
        for _, row in scaled_data.iterrows():
            for node_name, node in self.nodes.items():
                if not node['parents']:
                    log_likelihood += node['distribution'].logpdf(row[node_name])
                else:
                    parent_values = row[node['parents']].values
                    log_likelihood += np.log(node['distribution'](parent_values).pdf(row[node_name]))
        return log_likelihood

    def cross_validate(self, data: pd.DataFrame, k_folds: int = 5) -> Tuple[float, float]:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        log_likelihoods = []

        for train_index, test_index in kf.split(data):
            train_data, test_data = data.iloc[train_index], data.iloc[test_index]
            
            self.fit(train_data)
            log_likelihood = self.log_likelihood(test_data)
            log_likelihoods.append(log_likelihood)
        
        return np.mean(log_likelihoods), np.std(log_likelihoods)