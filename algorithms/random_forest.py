import numpy as np
from sklearn.ensemble import RandomForestRegressor

from algorithms.abstract_algorithm import AbstractAlgorithm


class RandomForest(AbstractAlgorithm):
    def __init__(self):
        self.model = None

    def get_name(self):
        return "RF"

    def train(self, X, Y):
        assert(Y.shape[1] == 1)
        self.model = RandomForestRegressor(random_state=42, n_jobs=-1)  # use all processors
        self.model.fit(X, Y.squeeze())

    def predict(self, X):
        pred = self.model.predict(X)
        # TODO: unfortunately sklearn does not provide a variance estimate -- damn!
        unc = np.zeros(pred.shape)
        return pred, unc
