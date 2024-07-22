import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

class BayesianNode:
    def __init__(self, name):
        self.name = name
        self.parents = []
        self.children = []
        self.distribution = None
        self.params = {}
        self.regression_model = None

    def to_dict(self):
        return {
            'name': self.name,
            'parents': [parent.name for parent in self.parents],
            'children': [child.name for child in self.children],
            'distribution': str(self.distribution) if self.distribution else None,
            'params': self.params,
            'regression_model': self.regression_model.get_params() if self.regression_model else None
        }

    @classmethod
    def from_dict(cls, data):
        node = cls(data['name'])
        node.params = data['params']
        if data['regression_model']:
            node.regression_model = LinearRegression()
            node.regression_model.set_params(**data['regression_model'])
        if data['distribution']:
            node.distribution = getattr(stats, data['distribution'].split('.')[-1])
        return node

    def fit(self, data, parent_data=None):
        if parent_data is None or len(parent_data) == 0:
            mean, std = np.mean(data), np.std(data)
            self.distribution = stats.norm(loc=mean, scale=std)
            self.params = {"loc": mean, "scale": std}
        else:
            self.regression_model = LinearRegression()
            self.regression_model.fit(parent_data, data)
            residuals = data - self.regression_model.predict(parent_data)
            std = np.std(residuals)
            self.params = {
                "coefficients": self.regression_model.coef_,
                "intercept": self.regression_model.intercept_,
                "scale": std,
            }
        return self  # Return self to allow method chaining

    def log_probability(self, value, parent_values=None):
        if self.distribution is None and self.regression_model is None:
            raise ValueError("Distribution not fitted yet")

        if parent_values is None or len(parent_values) == 0:
            return self.distribution.logpdf(value)
        else:
            predicted = self.regression_model.predict([parent_values])[0]
            return stats.norm(loc=predicted, scale=self.params["scale"]).logpdf(value)

    def sample(self, size=1, parent_values=None):
        if self.distribution is None and self.regression_model is None:
            raise ValueError("Node not fitted yet")

        if parent_values is None or len(parent_values) == 0:
            return self.distribution.rvs(size=size)
        else:
            if self.regression_model is None:
                raise ValueError("Regression model not fitted for node with parents")
            predicted = self.regression_model.predict([parent_values])[0]
            return stats.norm(loc=predicted, scale=self.params["scale"]).rvs(size=size)

class CategoricalNode(BayesianNode):
    def __init__(self, name, categories):
        super().__init__(name)
        self.categories = list(range(len(categories)))  # Use integer codes
        self.original_categories = categories
        self.distribution = stats.multinomial
        self.cpt = None

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'categories': self.categories,
            'original_categories': self.original_categories,
            'cpt': self.cpt.tolist() if self.cpt is not None else None
        })
        return base_dict

    @classmethod
    def from_dict(cls, data):
        node = cls(data['name'], data['original_categories'])
        node.params = data['params']
        node.categories = data['categories']
        node.cpt = np.array(data['cpt']) if data['cpt'] is not None else None
        return node

    def fit(self, data, parent_data=None):
        if parent_data is None or len(parent_data) == 0:
            counts = np.bincount(data, minlength=len(self.categories))
            self.distribution = stats.multinomial(n=1, p=counts / np.sum(counts))
            self.params = {"p": self.distribution.p}
            self.cpt = self.params["p"]
        else:
            parent_combinations = np.array(np.meshgrid(*[range(len(set(parent_data[col]))) for col in parent_data.columns])).T.reshape(-1, parent_data.shape[1])
            
            self.cpt = np.zeros((len(parent_combinations), len(self.categories)))
            for i, parent_comb in enumerate(parent_combinations):
                mask = np.all(parent_data == parent_comb, axis=1)
                counts = np.bincount(data[mask], minlength=len(self.categories))
                self.cpt[i] = counts / np.sum(counts)

            self.params = {"cpt": self.cpt}
        return self  # Return self to allow method chaining

    def log_probability(self, value, parent_values=None):
        if self.cpt is None:
            raise ValueError("Distribution not fitted yet")

        if parent_values is None or len(parent_values) == 0:
            return np.log(self.cpt[value])
        else:
            parent_index = np.ravel_multi_index(parent_values, [len(set(parent_values[i])) for i in range(len(parent_values))])
            return np.log(self.cpt[parent_index, value])

    def sample(self, size=1, parent_samples=None):
        probs = self.get_conditional_probabilities(parent_samples)
        return np.random.choice(self.categories, size=size, p=probs)

    def get_conditional_probabilities(self, parent_samples=None):
        if parent_samples is None or len(parent_samples) == 0:
            return self.cpt
        else:
            parent_index = np.ravel_multi_index(parent_samples, [len(set(parent_samples[i])) for i in range(len(parent_samples))])
            return self.cpt[parent_index]