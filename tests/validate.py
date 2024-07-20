import unittest
import pandas as pd
import numpy as np
from bayesian_network import BayesianNetwork

class TestBayesianNetwork(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'A': np.random.normal(0, 1, 100),
            'B': np.random.normal(0, 1, 100),
            'C': np.random.normal(0, 1, 100)
        })
        self.bn = BayesianNetwork(method='hill_climb', max_parents=2)

    def test_fit(self):
        self.bn.fit(self.data)
        self.assertIsNotNone(self.bn.nodes)
        self.assertGreater(len(self.bn.nodes), 0)

    def test_log_likelihood(self):
        self.bn.fit(self.data)
        ll = self.bn.log_likelihood(self.data)
        self.assertIsInstance(ll, float)

    def test_sample_node(self):
        self.bn.fit(self.data)
        sample = self.bn.sample_node('A', size=10)
        self.assertEqual(len(sample), 10)

    def test_cross_validate(self):
        mean, std = self.bn.cross_validate(self.data, k_folds=2)
        self.assertIsInstance(mean, float)
        self.assertIsInstance(std, float)
     
    def test_log_likelihood(self):
        self.bn.fit(self.data)
        ll = self.bn.log_likelihood(self.data)
        self.assertIsInstance(ll, float)
        self.assertTrue(np.isfinite(ll))  # Check if the log-likelihood is a finite number    
    
if __name__ == '__main__':
    unittest.main()