import numpy as np
from sklearn.mixture import GaussianMixture
from gmr.gmm import GMM
from gmr.sklearn import GaussianMixtureRegressor
from algorithms.abstract_algorithm import AbstractAlgorithm


class GMMRegression(AbstractAlgorithm):
    def __init__(self, n_components: int=2):
        self.model = None
        self.optimize = False
        self.n_components = n_components

    def get_name(self):
        return f"GMMRegression_n{self.n_components}"

    def train(self, X, Y):
        # gmm = GaussianMixture(n_components=self.n_components, covariance_type="diag", random_state=42)
        # gmm.fit(X, Y)
        # covariances = [np.diag(_c) for _c in gmm.covariances_]
        # self.model = GMM(n_components=self.n_components, priors=gmm.weights_, means=gmm.means_, 
        #                 covariances=covariances, random_state=42)
        self.model = GaussianMixtureRegressor(n_components=self.n_components)
        self.model.fit(X, Y)

    def predict(self, X):
        # TODO: correct for variance computation
        # pred = self.model.predict(np.arange(X.shape[1]), X).reshape(-1, 1)
        pred = self.model.predict(X)
        unc = np.diag(self.model.gmm_.to_mvn().covariance).mean()
        return pred, np.repeat(unc, pred.shape[0])[:, np.newaxis]

    def predict_f(self, X):
        return self.predict(X)
