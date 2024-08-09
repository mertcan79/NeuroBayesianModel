from abc import ABC, abstractmethod

class BaseBayesianNetwork(ABC):
    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def compute_edge_probabilities(self):
        pass
