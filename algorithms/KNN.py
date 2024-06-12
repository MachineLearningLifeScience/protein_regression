import warnings
from typing import Tuple

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from skopt import gbrt_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

from algorithms.abstract_algorithm import AbstractAlgorithm


class KNN(AbstractAlgorithm):
    def __init__(self, optimize: bool = False, opt_budget: int = 75, seed=42) -> None:
        self.model = None
        self.optimize = optimize
        self.seed = seed
        self.opt_budget = opt_budget
        self.X = None
        self.n_cv_splits = 3

    def get_name(self) -> str:
        return "KNN"

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        assert Y.shape[1] == 1
        self.model = KNeighborsRegressor(
            n_neighbors=int(np.ceil(0.3 * len(X))), n_jobs=-1
        )  # use all processors
        self._X = X  # NOTE: book-keeping for later neighbor inquiry
        self._y = Y
        Y = Y.squeeze() if Y.shape[0] > 1 else Y
        if self.optimize:
            opt_space = [
                Integer(
                    1,
                    max(2, np.floor(0.95 * len(X) / self.n_cv_splits)),
                    name="n_neighbors",
                ),  # NOTE: max number of neighbors is 95% of available data (account for splitting into thirds) for optimization stability, specifically on TOXI
            ]

            @use_named_args(opt_space)
            def _opt_objective(**params):
                self.model.set_params(**params)
                return -np.mean(
                    cross_val_score(
                        self.model,
                        X,
                        Y,
                        cv=self.n_cv_splits,
                        n_jobs=1,
                        scoring="neg_mean_absolute_error",
                        error_score="raise",
                    )
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_gp = gbrt_minimize(
                    _opt_objective,
                    opt_space,
                    n_calls=self.opt_budget,
                    random_state=self.seed,
                )
            self.optimal_parameters = res_gp.x
            print(f"Score: {res_gp.fun}")
            print(f"Parameters: k={res_gp.x[0]}")
            self.model = KNeighborsRegressor(n_neighbors=res_gp.x[0], n_jobs=-1)
        self.model.fit(X, Y)

    def predict(self, X) -> Tuple[np.array, np.array]:
        """
        Returns:
            pred - model predictions
            unc - model STD across neighbors
        """
        pred = self.model.predict(X).reshape(-1, 1)
        queried_neighbor_values = self._y[
            self.model.kneighbors(X, return_distance=False)
        ]
        neighbor_means = np.mean(queried_neighbor_values, axis=1)
        np.testing.assert_almost_equal(
            neighbor_means, pred
        )  # NOTE: this fails if distances or weights are accounted for
        var = np.var(queried_neighbor_values, axis=1).reshape(-1, 1)
        assert pred.shape == var.shape
        return pred, var

    def predict_f(self, X: np.ndarray):
        return self.predict(X)
