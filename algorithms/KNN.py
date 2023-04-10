import numpy as np
from typing import Tuple
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from algorithms.abstract_algorithm import AbstractAlgorithm
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt import gp_minimize


class KNN(AbstractAlgorithm):
    def __init__(self, optimize: bool=False, k_max: int=100, opt_budget: int=100, seed=42) -> None:
        self.model = None
        self.optimize = optimize
        self.seed = seed
        if self.optimize:
            self.k_max = k_max
            self.opt_budget = opt_budget
            self.opt_space = [
            Integer(1, self.k_max, name="n_neighbors"),
        ]

    def get_name(self) -> str:
        return "KNN"

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        assert(Y.shape[1] == 1)
        self.model = KNeighborsRegressor(n_neighbors=int(np.ceil(0.3*len(X))), n_jobs=-1)  # use all processors
        Y = Y.squeeze() if Y.shape[0] > 1 else Y
        if self.optimize:
            self.k_max = int(len(X)) # all data is maximal possible 
            @use_named_args(self.opt_space)
            def _opt_objective(**params):
                self.model.set_params(**params)
                return -np.mean(cross_val_score(self.model, X, Y, cv=5, n_jobs=-1, scoring="neg_mean_absolute_error"))
            res_gp = gp_minimize(_opt_objective, self.opt_space, n_calls=self.opt_budget, random_state=self.seed)
            print(f"Score: {res_gp.fun}")
            print(f"Parameters: k={res_gp.x[0]}")
            self.model = KNeighborsRegressor(n_neighbors=res_gp.x[0], n_jobs=-1) 
        self.model.fit(X, Y)

    def predict(self, X) -> Tuple[np.array, np.array]:
        """
        Returns:
            pred - model predictions
            unc - model variance as E[(f(x) - E[f(x)])**2]
        """
        pred = self.model.predict(X).reshape(-1, 1)
        unc = np.mean(np.square(pred-np.mean(pred)), axis=1).reshape(-1, 1)
        assert pred.shape == unc.shape
        return pred, unc

    def predict_f(self, X: np.ndarray):
        return self.predict(X)





