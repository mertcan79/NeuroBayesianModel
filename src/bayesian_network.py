import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pgmpy.estimators import PC, HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesianNode:
    def __init__(self, name: str):
        self.name = name
        self.parents = []
        self.children = []
        self.distribution = None

    def set_distribution(self, distribution):
        self.distribution = distribution

class BayesianNetwork:
    def __init__(self, method='hill_climb', max_parents=3):
        self.method = method
        self.max_parents = max_parents
        self.nodes: Dict[str, BayesianNode] = {}
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

        if self.method == 'pc':
            pc = PC(data=scaled_data)
            skeleton, _ = pc.estimate(max_cond_vars=self.max_parents)
            self.model = skeleton.to_directed()
        elif self.method == 'hill_climb':
            hc = HillClimbSearch(data=scaled_data)
            self.model = hc.estimate(scoring_method=BicScore(data=scaled_data), max_indegree=self.max_parents)
        else:
            raise ValueError("Unsupported structure learning method")

        if prior_edges:
            for edge in prior_edges:
                if edge not in self.model.edges():
                    self.model.add_edge(edge[0], edge[1])

        self.nodes = {node: BayesianNode(node) for node in self.model.nodes()}
        for edge in self.model.edges():
            self.nodes[edge[1]].parents.append(self.nodes[edge[0]])
            self.nodes[edge[0]].children.append(self.nodes[edge[1]])

        logger.info("Learned graph structure:")
        for node_name, node in self.nodes.items():
            logger.info(f"{node_name} -> parents: {[p.name for p in node.parents]}, children: {[c.name for c in node.children]}")

    def _fit_parameters(self, data: pd.DataFrame):
        imputed_data = self.imputer.transform(data)
        scaled_data = self.scaler.transform(imputed_data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

        for node_name, node in self.nodes.items():
            node_data = scaled_data[node_name].values
            if not node.parents:
                mean, std = np.mean(node_data), np.std(node_data)
                node.set_distribution(stats.norm(loc=mean, scale=std))
            else:
                parent_data = scaled_data[[p.name for p in node.parents]].values
                beta = np.linalg.lstsq(parent_data, node_data, rcond=None)[0]
                residuals = node_data - parent_data.dot(beta)
                std = np.std(residuals)
                node.set_distribution(lambda x, beta=beta, std=std: stats.norm(loc=x.dot(beta), scale=std))

        logger.info("Fitted parameters")

    def sample_node(self, node_name: str, size: int = 1) -> np.ndarray:
        node = self.nodes[node_name]
        if not node.parents:
            samples = node.distribution.rvs(size=size)
        else:
            parent_values = np.array([self.sample_node(p.name, size) for p in node.parents]).T
            samples = node.distribution(parent_values).rvs(size=size)
        return self.scaler.inverse_transform(samples.reshape(-1, 1)).flatten()

    def log_likelihood(self, data: pd.DataFrame) -> float:
        imputed_data = self.imputer.transform(data)
        scaled_data = self.scaler.transform(imputed_data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

        log_likelihood = 0
        for _, row in scaled_data.iterrows():
            for node_name, node in self.nodes.items():
                if not node.parents:
                    log_likelihood += node.distribution.logpdf(row[node_name])
                else:
                    parent_values = row[[p.name for p in node.parents]].values
                    log_likelihood += np.log(node.distribution(parent_values).pdf(row[node_name]))
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