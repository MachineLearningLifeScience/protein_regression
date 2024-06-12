import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from skopt import gbrt_minimize
from skopt.space import Integer

from algorithms.abstract_algorithm import AbstractAlgorithm


class MLPEnsemble(AbstractAlgorithm):
    def __init__(
        self, optimize: bool = False, opt_budget: int = 10, seed=10, ensemble_size=3
    ) -> None:
        self.model = None
        self.optimize = optimize
        self.seed = seed
        self.ensemble_size = ensemble_size
        self.opt_budget = opt_budget
        self.X = None
        self.n_cv_splits = 3

    def get_name(self) -> str:
        return "MLP"

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        assert Y.shape[1] == 1
        self.model = [
            MLPRegressor(random_state=self.seed + _e)
            for _e in range(self.ensemble_size)
        ]
        raise NotImplementedError("NO OPTIMIZATION IMPLEMENTED")
        # TODO: optimzie all regressors

    def predict(self, X: np.ndarray):
        """
        Returns:
            pred - model predictions
            unc - model variance across ensemble
        """
        predictions = np.array(
            [model.predict(X).reshape(-1, 1) for model in self.model]
        )
        pred = np.mean(predictions, axis=1)  # TODO: testing here
        var = np.var(predictions, axis=1).reshape(-1, 1)
        assert pred.shape == var.shape
        return pred, var

    def predict_f(self, X: np.ndarray):
        return self.predict(X)
