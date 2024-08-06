class EEGPreprocessor:
    def __init__(self, filters=None, normalize=True):
        self.filters = filters or []
        self.normalize = normalize

    def fit(self, X, y=None):
        if self.normalize:
            self.mean = X.mean(axis=1, keepdims=True)
            self.std = X.std(axis=1, keepdims=True)
        return self

    def transform(self, X):
        for filter_func in self.filters:
            X = filter_func(X)
        if self.normalize:
            X = (X - self.mean) / self.std
        return X