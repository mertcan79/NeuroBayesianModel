import pytest
from scipy import stats
import sys
import os

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from bayesian_node import BayesianNode

def test_bayesian_node_creation():
    node = BayesianNode("test", stats.norm(0, 1))
    assert node.name == "test"
    assert isinstance(node.distribution, stats.norm)

def test_bayesian_node_update():
    node = BayesianNode("test", stats.norm(0, 1))
    data = [1, 2, 3, 4, 5]
    node.update(data)
    assert len(node.data) == 5
    assert node.distribution.mean() != 0  # The mean should have changed

# More tests...