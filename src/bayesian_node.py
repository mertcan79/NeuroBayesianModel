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

    def transform(self, data):
        return data  # For continuous variables, no transformation is needed

    def fit(self, data, parent_data=None):
        if parent_data is None or len(parent_data) == 0:
            # Fit univariate normal distribution
            mean, std = stats.norm.fit(data)
            self.distribution = stats.norm(loc=mean, scale=std)
            self.params = {'loc': mean, 'scale': std}
        else:
            # Fit multivariate linear regression
            self.regression_model = LinearRegression()
            self.regression_model.fit(parent_data, data)
            residuals = data - self.regression_model.predict(parent_data)
            _, std = stats.norm.fit(residuals)
            self.params = {
                'coefficients': self.regression_model.coef_,
                'intercept': self.regression_model.intercept_,
                'scale': std
            }

    def log_probability(self, value, parent_values=None):
        if self.distribution is None and self.regression_model is None:
            raise ValueError("Distribution not fitted yet")
        
        if parent_values is None or len(parent_values) == 0:
            return self.distribution.logpdf(value)
        else:
            predicted = self.regression_model.predict([parent_values])[0]
            return stats.norm(loc=predicted, scale=self.params['scale']).logpdf(value)

    def sample(self, size=1, parent_values=None):
        if self.distribution is None and self.regression_model is None:
            raise ValueError("Distribution not fitted yet")
        
        if parent_values is None or len(parent_values) == 0:
            return self.distribution.rvs(size=size)
        else:
            predicted = self.regression_model.predict([parent_values])[0]
            return stats.norm(loc=predicted, scale=self.params['scale']).rvs(size=size)

class CategoricalNode:
    def __init__(self, name, categories):
        self.name = name
        self.categories = categories
        self.category_map = {cat: i for i, cat in enumerate(categories)}
        self.reverse_category_map = {i: cat for i, cat in enumerate(categories)}
        self.distribution = None
        self.params = {}
        self.cpt = None
        self.parents = []
        self.children = []

    def transform(self, data):
        if np.isscalar(data):
            return self.category_map.get(data, -1)
        else:
            return np.array([self.category_map.get(d, -1) for d in data])

    def fit(self, data, parent_data=None):
        if parent_data is None or len(parent_data) == 0:
            # Fit categorical distribution
            counts = np.bincount(data, minlength=len(self.categories))
            self.distribution = stats.multinomial(n=1, p=counts / np.sum(counts))
            self.params = {'p': self.distribution.p}
            self.cpt = self.params['p']
        else:
            # Fit conditional probability table
            parent_categories = [parent.categories for parent in self.parents]
            parent_combinations = np.array(np.meshgrid(*parent_categories)).T.reshape(-1, len(self.parents))
            
            self.cpt = np.zeros((len(parent_combinations), len(self.categories)))
            for i, parent_comb in enumerate(parent_combinations):
                mask = np.all(parent_data == parent_comb, axis=1)
                counts = np.bincount(data[mask], minlength=len(self.categories))
                self.cpt[i] = counts / np.sum(counts)
            
            self.params = {'cpt': self.cpt}

    def log_probability(self, value, parent_values=None):
        if self.cpt is None:
            raise ValueError("Distribution not fitted yet")
        
        value_index = self.transform(value)
        
        if parent_values is None or len(parent_values) == 0:
            return np.log(self.cpt[value_index])
        else:
            parent_indices = [parent.transform(pv) for parent, pv in zip(self.parents, parent_values)]
            parent_index = np.ravel_multi_index(parent_indices, [len(parent.categories) for parent in self.parents])
            return np.log(self.cpt[parent_index, value_index])

    def sample(self, size=1, parent_values=None):
        if self.cpt is None:
            raise ValueError("Distribution not fitted yet")
        
        if parent_values is None or len(parent_values) == 0:
            probs = self.cpt
        else:
            parent_indices = [parent.transform(pv) for parent, pv in zip(self.parents, parent_values)]
            parent_index = np.ravel_multi_index(parent_indices, [len(parent.categories) for parent in self.parents])
            probs = self.cpt[parent_index]
        
        samples = np.random.choice(len(self.categories), size=size, p=probs)
        return np.array([self.reverse_category_map[s] for s in samples])