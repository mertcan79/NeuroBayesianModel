import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple
from bayesian_node import BayesianNode
from pgmpy.estimators import MmhcEstimator, HillClimbSearch, BayesianEstimator, BDeuScore
from pgmpy.models import BayesianNetwork as PgmpyBN
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel



class BayesianNetwork:
    def __init__(self):
        self.nodes: Dict[str, BayesianNode] = {}
        self.graph = nx.DiGraph()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def add_node(self, node: BayesianNode):
        self.nodes[node.name] = node
        self.graph.add_node(node.name)
        for parent in node.parents:
            self.graph.add_edge(parent.name, node.name)

    def learn_structure(self, data: pd.DataFrame, prior_edges: List[tuple] = None, method: str = 'hill_climb', max_parents: int = 3):
        # Preprocess data
        imputed_data = self.imputer.fit_transform(data)
        scaled_data = self.scaler.fit_transform(imputed_data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

        if method == 'hill_climb':
            hc = HillClimbSearch(scaled_data)
            scoring_method = BicScore(data=scaled_data)
            scoring_method.max_parents = max_parents
            est_model = hc.estimate(scoring_method=scoring_method)
        else:
            raise ValueError("Unsupported structure learning method")

        # Convert the estimated model to a BayesianModel
        model = BayesianModel(est_model.edges())

        # Incorporate prior knowledge if provided
        if prior_edges:
            for edge in prior_edges:
                if edge not in model.edges():
                    model.add_edge(edge[0], edge[1])

        self.graph = model.to_directed()

        # Update nodes based on learned structure
        self.nodes = {col: BayesianNode(col, None) for col in data.columns}
        for edge in self.graph.edges():
            self.nodes[edge[1]].parents.append(self.nodes[edge[0]])
            self.nodes[edge[0]].children.append(self.nodes[edge[1]])

        print("Learned graph structure:")
        for node, neighbors in self.graph.adj.items():
            print(f"{node} -> {list(neighbors.keys())}")

        # Update nodes based on learned structure
        self.nodes = {col: BayesianNode(col, None) for col in data.columns}
        for edge in self.graph.edges():
            self.nodes[edge[1]].parents.append(self.nodes[edge[0]])
            self.nodes[edge[0]].children.append(self.nodes[edge[1]])

        print("Learned graph structure:")
        for node, neighbors in self.graph.adj.items():
            print(f"{node} -> {list(neighbors.keys())}")

    def fit(self, data: pd.DataFrame):
        imputed_data = self.imputer.transform(data)
        scaled_data = self.scaler.transform(imputed_data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

        for node_name, node in self.nodes.items():
            if not node.parents:
                # For nodes without parents, fit a univariate distribution
                node_data = scaled_data[node_name].values
                mean, std = np.mean(node_data), np.std(node_data)
                node.distribution = stats.norm(loc=mean, scale=std)
            else:
                # For nodes with parents, fit a Gaussian Process
                parent_names = [parent.name for parent in node.parents]
                X = scaled_data[parent_names].values
                y = scaled_data[node_name].values
                
                kernel = RBF() + WhiteKernel() + Matern()
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
                gp.fit(X, y)
                
                node.distribution = lambda x, gp=gp: stats.norm(
                    loc=gp.predict(x.reshape(1, -1) if x.ndim == 1 else x)[0],
                    scale=np.sqrt(gp.predict(x.reshape(1, -1) if x.ndim == 1 else x, return_std=True)[1][0])
                )

        print("Fitted parameters:")
        for node_name, node in self.nodes.items():
            if callable(node.distribution):
                print(f"{node_name}: Gaussian Process (non-linear conditional distribution)")
            else:
                print(f"{node_name}: Mean = {node.distribution.mean():.4f}, Std = {node.distribution.std():.4f}")

    def sample_node(self, node_name: str, size: int = 1) -> np.ndarray:
        node = self.nodes[node_name]
        if node.parents:
            parent_values = np.array([self.sample_node(parent.name, size) for parent in node.parents]).T
            samples = node.distribution(parent_values).rvs(size=size)
        else:
            samples = node.distribution.rvs(size=size)
        return self.scaler.inverse_transform(samples.reshape(-1, 1)).flatten()

    def copy(self) -> 'BayesianNetwork':
        new_network = BayesianNetwork()
        new_network.nodes = {name: node.copy() for name, node in self.nodes.items()}
        new_network.graph = self.graph.copy()
        new_network.scaler = self.scaler
        new_network.imputer = self.imputer
        return new_network

    def cross_validate(self, data: pd.DataFrame, k_folds: int = 5) -> Tuple[float, float]:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        log_likelihoods = []

        for train_index, test_index in kf.split(data):
            train_data, test_data = data.iloc[train_index], data.iloc[test_index]
            
            self.learn_structure(train_data)
            self.fit(train_data)
            
            log_likelihood = self.log_likelihood(test_data)
            log_likelihoods.append(log_likelihood)
        
        return np.mean(log_likelihoods), np.std(log_likelihoods)

    def log_likelihood(self, data: pd.DataFrame) -> float:
        imputed_data = self.imputer.transform(data)
        scaled_data = self.scaler.transform(imputed_data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

        log_likelihood = 0
        for _, row in scaled_data.iterrows():
            for node_name, node in self.nodes.items():
                if node.parents:
                    parent_values = row[[parent.name for parent in node.parents]].values
                    log_likelihood += np.log(node.distribution(parent_values).pdf(row[node_name]))
                else:
                    log_likelihood += node.distribution.logpdf(row[node_name])
        return log_likelihood