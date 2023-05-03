import warnings
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from algorithms.abstract_algorithm import AbstractAlgorithm
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt import gbrt_minimize


class RandomForest(AbstractAlgorithm):
    def __init__(self, optimize: bool=False, seed: int=42, opt_budget: int=15, persist_optimal_parameters: str=None):
        self.model = None
        self.optimize = optimize
        self.seed = seed
        self.opt_budget = opt_budget
        self.model = RandomForestRegressor(random_state=self.seed, n_jobs=-1)  # use all processors
        self._persist_id = persist_optimal_parameters

    def get_name(self):
        return "RF"

    def train(self, X, Y):
        assert(Y.shape[1] == 1)
        Y = Y.ravel() if Y.shape[0] > 1 else Y
        if self.optimize:
            opt_space = [
                Integer(2, len(X), name="n_estimators"), 
            ]
            @use_named_args(opt_space)
            def _opt_objective(**params):
                self.model.set_params(**params)
                return -np.mean(cross_val_score(self.model, X, Y, cv=3, n_jobs=1, scoring="neg_mean_absolute_error", error_score="raise"))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_gp = gbrt_minimize(_opt_objective, opt_space, n_calls=self.opt_budget, random_state=self.seed)
            self.optimal_parameters = res_gp.x
            print(f"Score: {res_gp.fun}")
            print(f"Parameters: N={res_gp.x[0]}")
            self.model = RandomForestRegressor(
                        n_estimators=res_gp.x[0],
                        random_state=self.seed, 
                        n_jobs=-1,
                        )
            if self._persist_id:
                # NOTE: use hash, data, splitter, split when persisting optimal parameters
                filename = f"./results/cache/RF_optimal_estimators_{self._persist_id}.pkl"
                with open(filename, "wb") as outfile:
                    pickle.dump(dict(n_estimators=res_gp.x[0]), outfile)
        self.model.fit(X, Y.ravel())

    def predict(self, X):
        pred = self.model.predict(X).reshape(-1, 1)
        unc = np.zeros(pred.shape)
        return pred, unc

    def predict_f(self, X):
        return self.predict(X)
