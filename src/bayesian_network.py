import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
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
    def __init__(self, method='hill_climb', max_parents=3):
        self.method = method
        self.max_parents = max_parents
        self.nodes: Dict[str, Any] = {}
        self.graph = nx.DiGraph()
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
        # Preprocessing
        imputed_data = self.imputer.fit_transform(data)
        scaled_data = self.scaler.fit_transform(imputed_data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

        # Structure learning
        if self.method == 'hill_climb':
            hc = HillClimbSearch(data=scaled_data)
            est_model = hc.estimate(scoring_method=BicScore(data=scaled_data), max_indegree=self.max_parents)
        else:
            raise ValueError("Unsupported structure learning method")

        model = BayesianModel(est_model.edges())

        if prior_edges:
            for edge in prior_edges:
                if edge not in model.edges():
                    model.add_edge(edge[0], edge[1])

        self.graph = model.to_directed()

        self.nodes = {col: {'parents': [], 'children': []} for col in data.columns}
        for edge in self.graph.edges():
            self.nodes[edge[1]]['parents'].append(edge[0])
            self.nodes[edge[0]]['children'].append(edge[1])

        logger.info("Learned graph structure:")
        for node, neighbors in self.graph.adj.items():
            logger.info(f"{node} -> {list(neighbors.keys())}")

    def _fit_parameters(self, data: pd.DataFrame):
        imputed_data = self.imputer.transform(data)
        scaled_data = self.scaler.transform(imputed_data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

        for node_name, node in self.nodes.items():
            if not node['parents']:
                # Fit marginal distribution
                node_data = scaled_data[node_name].values
                mean, std = np.mean(node_data), np.std(node_data)
                node['distribution'] = stats.norm(loc=mean, scale=std)
            else:
                # Use simple linear regression as an alternative
                from sklearn.linear_model import LinearRegression
                X = scaled_data[node['parents']].values
                y = scaled_data[node_name].values
                model = LinearRegression()
                model.fit(X, y)
                node['distribution'] = lambda x, model=model: stats.norm(
                    loc=model.predict(x.reshape(1, -1) if x.ndim == 1 else x),
                    scale=0.1  # Placeholder standard deviation
                )

        logger.info("Fitted parameters:")
        for node_name, node in self.nodes.items():
            if callable(node['distribution']):
                logger.info(f"{node_name}: Linear Regression (conditional distribution)")
            else:
                logger.info(f"{node_name}: Mean = {node['distribution'].mean():.4f}, Std = {node['distribution'].std():.4f}")

    def sample_node(self, node_name: str, size: int = 1) -> np.ndarray:
        node = self.nodes[node_name]
        if node['parents']:
            parent_values = np.array([self.sample_node(parent, size) for parent in node['parents']]).T
            samples = node['distribution'](parent_values).rvs(size=size)
        else:
            samples = node['distribution'].rvs(size=size)
        return self.scaler.inverse_transform(samples.reshape(-1, 1)).flatten()

    def log_likelihood(self, data: pd.DataFrame) -> float:
        imputed_data = self.imputer.transform(data)
        scaled_data = self.scaler.transform(imputed_data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

        log_likelihood = 0
        for _, row in scaled_data.iterrows():
            for node_name, node in self.nodes.items():
                if node['parents']:
                    parent_values = row[node['parents']].values
                    log_likelihood += np.log(node['distribution'](parent_values).pdf(row[node_name]))
                else:
                    log_likelihood += node['distribution'].logpdf(row[node_name])
        return log_likelihood

    def visualize(self, filename: str = None):
        plt.figure(figsize=(12, 8))
        nx.draw(self.graph, with_labels=True, node_color='lightblue', 
                node_size=3000, font_size=12, font_weight='bold')
        plt.title("Bayesian Network Structure")
        if filename:
            plt.savefig(filename)
        plt.show()

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
