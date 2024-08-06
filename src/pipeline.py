class EEGModelPipeline:
    def __init__(self, preprocessor, model, augmenter=None):
        self.preprocessor = preprocessor
        self.model = model
        self.augmenter = augmenter

    def fit(self, X, y):
        X_prep = self.preprocessor.fit_transform(X)
        if self.augmenter:
            X_aug, y_aug = self.augmenter.augment(X_prep, y)
            self.model.fit(X_aug, y_aug)
        else:
            self.model.fit(X_prep, y)
        return self

    def predict(self, X):
        X_prep = self.preprocessor.transform(X)
        return self.model.predict(X_prep)