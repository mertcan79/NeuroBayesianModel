from scipy import stats
import numpy as np

def test_sample_node():
    class SimpleNode:
        def __init__(self, name):
            self.name = name
            self.distribution = stats.norm(loc=0, scale=1)

        def sample(self, size=1):
            return self.distribution.rvs(size=size)

    class SimpleBayesianNetwork:
        def __init__(self):
            self.nodes = {
                'A': SimpleNode('A'),
                'B': SimpleNode('B')
            }

        def sample_node(self, node_name: str, size: int = 1) -> np.ndarray:
            print(f"sample_node called with node_name: {node_name}, size: {size}")
            node = self.nodes[node_name]
            return node.sample(size)

    bn = SimpleBayesianNetwork()
    try:
        print(bn.sample_node('A', size=5))  # Test sampling
    except Exception as e:
        print(f"Error: {e}")

test_sample_node()