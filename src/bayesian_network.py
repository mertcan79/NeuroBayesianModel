import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pgmpy.estimators import PC, HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
import logging
import time

logging.basicConfig(level=logging.DEBUG)  # Change to DEBUG for more verbose logging
logger = logging.getLogger(__name__)

class BayesianNode:
    def __init__(self, name: str):
        self.name = name
        self.parents = []
        self.children = []
        self.distribution = None
        self.params = {}
        self.scaler = StandardScaler()

    def set_distribution(self, distribution, params=None):
        self.distribution = distribution
        self.params = params

    def fit_scaler(self, data):
        logger.debug(f"Fitting scaler for node {self.name} with data: {data}")
        self.scaler.fit(data.reshape(-1, 1))

    def transform(self, data):
        transformed = self.scaler.transform(data.reshape(-1, 1)).flatten()
        logger.debug(f"Transforming data for node {self.name}: {data} -> {transformed}")
        return transformed

    def inverse_transform(self, data):
        inversed = self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        logger.debug(f"Inverse transforming data for node {self.name}: {data} -> {inversed}")
        return inversed

class BayesianNetwork:
    def __init__(self, method='hill_climb', max_parents=3):
        self.method = method
        self.max_parents = max_parents
        self.nodes: Dict[str, BayesianNode] = {}
        self.model = None
        self.imputer = SimpleImputer(strategy='mean')

    def fit(self, data: pd.DataFrame, prior_edges: List[tuple] = None, progress_callback: Callable[[float], None] = None):
        try:
            logger.info("Starting to learn structure")
            self._learn_structure(data, prior_edges)
            if progress_callback:
                progress_callback(0.5)
            logger.info("Fitting parameters")
            self._fit_parameters(data)
            if progress_callback:
                progress_callback(1.0)
        except Exception as e:
            logger.error(f"Error during fitting: {str(e)}")
            raise

    def _learn_structure(self, data: pd.DataFrame, prior_edges: List[tuple] = None):
        imputed_data = self.imputer.fit_transform(data)
        imputed_data = pd.DataFrame(imputed_data, columns=data.columns)
        logger.debug(f"Imputed data for structure learning:\n{imputed_data.head()}")

        if self.method == 'pc':
            pc = PC(data=imputed_data)
            skeleton, _ = pc.estimate(max_cond_vars=self.max_parents)
            self.model = skeleton.to_directed()
        elif self.method == 'hill_climb':
            hc = HillClimbSearch(data=imputed_data)
            self.model = hc.estimate(scoring_method=BicScore(data=imputed_data), max_indegree=self.max_parents)
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
        imputed_data = pd.DataFrame(imputed_data, columns=data.columns)
        logger.debug(f"Imputed data for parameter fitting:\n{imputed_data.head()}")

        for node_name, node in self.nodes.items():
            node_data = imputed_data[node_name].values
            node.fit_scaler(node_data)
            scaled_node_data = node.transform(node_data)

            logger.debug(f"Node {node_name} - Scaled data: {scaled_node_data}")

            if not node.parents:
                mean, std = np.mean(scaled_node_data), np.std(scaled_node_data)
                node.set_distribution(stats.norm, params={'loc': mean, 'scale': std})
                logger.debug(f"Node {node_name} - Distribution parameters (mean, std): {mean}, {std}")
            else:
                parent_data = np.column_stack([self.nodes[p.name].transform(imputed_data[p.name].values) for p in node.parents])
                beta = np.linalg.lstsq(parent_data, scaled_node_data, rcond=None)[0]
                residuals = scaled_node_data - parent_data.dot(beta)
                std = np.std(residuals)
                node.set_distribution(stats.norm, params={'beta': beta, 'scale': std})
                logger.debug(f"Node {node_name} - Parameters (beta, std): {beta}, {std}")

        logger.info("Fitted parameters")

    def log_likelihood(self, data: pd.DataFrame) -> float:
        imputed_data = self.imputer.transform(data)
        imputed_data = pd.DataFrame(imputed_data, columns=data.columns)
        logger.debug(f"Imputed data for log-likelihood calculation:\n{imputed_data.head()}")

        log_likelihood = 0.0
        for _, row in imputed_data.iterrows():
            for node_name, node in self.nodes.items():
                scaled_value = node.transform(row[node_name])
                logger.debug(f"Node {node_name} - Scaled value: {scaled_value}")

                if not node.parents:
                    # Node without parents
                    log_likelihood += node.distribution.logpdf(scaled_value, **node.params)
                else:
                    # Node with parents
                    parent_values = np.array([self.nodes[p.name].transform(row[p.name]) for p in node.parents])
                    logger.debug(f"Node {node_name} - Parent values shape: {parent_values.shape}")
                    logger.debug(f"Node {node_name} - Beta shape: {node.params['beta'].shape}")
                    
                    # Ensure correct shape for beta and parent_values
                    if parent_values.ndim == 1:
                        parent_values = parent_values.reshape(-1, 1)  # Reshape to (n, 1)
                    
                    loc = np.dot(parent_values.T, node.params['beta'])  # Transpose parent_values for dot product
                    scale = node.params['scale']
                    logger.debug(f"Node {node_name} - Location: {loc}, Scale: {scale}")
                    
                    log_likelihood += node.distribution.logpdf(scaled_value, loc=loc, scale=scale)

        logger.info(f"Total log-likelihood: {log_likelihood}")
        return float(log_likelihood)

    def cross_validate(self, data: pd.DataFrame, k_folds: int = 5, timeout: int = 300) -> Tuple[float, float]:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        log_likelihoods = []

        start_time = time.time()
        for i, (train_index, test_index) in enumerate(kf.split(data)):
            if time.time() - start_time > timeout:
                logger.warning(f"Cross-validation timed out after {timeout} seconds")
                break

            logger.info(f"Starting fold {i+1}/{k_folds}")
            train_data, test_data = data.iloc[train_index], data.iloc[test_index]
            
            self.fit(train_data)
            log_likelihood = self.log_likelihood(test_data)
            log_likelihoods.append(log_likelihood)
            logger.info(f"Completed fold {i+1}/{k_folds}, log-likelihood: {log_likelihood}")

        if len(log_likelihoods) == 0:
            logger.error("No folds completed in cross-validation")
            return float('nan'), float('nan')
        
        return float(np.mean(log_likelihoods)), float(np.std(log_likelihoods))

    def sample_node(self, node_name: str, size: int = 1) -> np.ndarray:
        node = self.nodes[node_name]
        if not node.parents:
            samples = node.distribution.rvs(size=size, **node.params)
        else:
            parent_values = np.column_stack([self.sample_node(p.name, size) for p in node.parents])
            parent_values_scaled = np.column_stack([self.nodes[p.name].transform(parent_values[:, i]) for i, p in enumerate(node.parents)])
            loc = np.dot(parent_values_scaled, node.params['beta'])
            samples = node.distribution.rvs(loc=loc, scale=node.params['scale'], size=size)
        return node.inverse_transform(samples.reshape(-1, 1))
