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
        self.model = GaussianMixtureRegressor(n_components=self.n_components, init_params="kmeans++")
        self.model.fit(X, Y)

    def predict(self, X):
        pred = self.model.predict(X)
        # sigma_2 = np.repeat(np.var(pred), X.shape[0])[:, np.newaxis]
        cluster_vec = np.argmax(self.model.gmm_.to_responsibilities(np.hstack([X, pred])), axis=-1)
        sigma_2 = []
        for cluster in cluster_vec:
            # NOTE: computing uncertainty sigma relative to Gaussian of MM: weight * (tr(\sigma)/2)
            _sigma = (np.trace(self.model.gmm_.covariances[cluster])/self.n_components) * self.model.gmm_.priors[cluster]
            sigma_2.append(_sigma)
        sigma_2 = np.array(sigma_2)[:, np.newaxis]
        return pred, sigma_2

    def predict_f(self, X):
        return self.predict(X)
