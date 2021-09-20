import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from algorithms.abstract_algorithm import AbstractAlgorithm


class KNN(AbstractAlgorithm):
    def __init__(self):
        self.model = None

    def get_name(self):
        return "KNN"

    def train(self, X, Y):
        assert(Y.shape[1] == 1)
        self.model = KNeighborsRegressor(n_neighbors=int(0.3*len(X)), n_jobs=-1)  # use all processors
        self.model.fit(X, Y.squeeze())

    def predict(self, X):
        pred = self.model.predict(X).reshape(-1, 1)
        unc = np.zeros(pred.shape)
        return pred, unc
