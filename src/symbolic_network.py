from hierarchical_network import HierarchicalBayesianNetwork
import numpy as np
import jax.numpy as jnp
import numpyro
import networkx as nx

class SymbolicBayesianNetwork(HierarchicalBayesianNetwork):
    def __init__(self, num_features, max_parents=2, iterations=50, *args, **kwargs):
        super().__init__(num_features, max_parents, iterations, *args, **kwargs)
        self.symbolic_rules = []
        self.causal_graph = nx.DiGraph()

    def add_symbolic_rule(self, rule):
        self.symbolic_rules.append(rule)

    def evaluate_symbolic_rule(self, rule, data, params):
        if rule['type'] == 'threshold':
            condition = data[:, self.feature_names.index(rule['feature'])] > rule['threshold']
            return jnp.where(condition, 0.0, -1e10)
        elif rule['type'] == 'relationship':
            var1 = params[rule['var1']]
            var2 = params[rule['var2']]
            return jnp.where(rule['operator'](var1, var2), 0.0, -1e10)
        elif rule['type'] == 'custom':
            return rule['function'](data, params)

    def fit(self, data):
        super().fit(data)
        for rule in self.symbolic_rules:
            numpyro.factor("rule_constraint", self.evaluate_symbolic_rule(rule, data, self.samples))
        self.learn_causal_structure()
        return self.samples

    def learn_causal_structure(self):
        self.causal_graph = nx.DiGraph()
        for i, parent in enumerate(self.feature_names):
            for j, child in enumerate(self.feature_names[i+1:], start=i+1):
                if abs(self.edge_weights[i][j]) > 0.5:
                    self.causal_graph.add_edge(parent, child)

    def perform_causal_inference(self, intervention, target):
        if not nx.has_path(self.causal_graph, intervention[0], target):
            return "No causal path found"
        intervention_effect = self.predict({intervention[0]: intervention[1]})
        baseline = self.predict({})
        return intervention_effect[target] - baseline[target]

    def analyze_feature_importance(self):
        importance = np.abs(self.edge_weights).sum(axis=1)
        return dict(zip(self.feature_names, importance))

    def generate_symbolic_insights(self, data):
        insights = []
        for rule in self.symbolic_rules:
            if rule['type'] == 'threshold':
                feature = rule['feature']
                threshold = rule['threshold']
                above_threshold = (data[feature] > threshold).mean()
                insights.append(f"{feature} is above {threshold} in {above_threshold:.2%} of cases")
            elif rule['type'] == 'relationship':
                var1, var2 = rule['var1'], rule['var2']
                operator = rule['operator'].__name__
                satisfaction = operator(data[var1], data[var2]).mean()
                insights.append(f"The relationship {var1} {operator} {var2} is satisfied in {satisfaction:.2%} of cases")
        return insights