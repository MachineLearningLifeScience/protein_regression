import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from algorithms.abstract_algorithm import AbstractAlgorithm
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize


class RandomForest(AbstractAlgorithm):
    def __init__(self, optimize=False, seed=42, opt_budget=100):
        self.model = None
        self.optimize = optimize
        self.seed = seed
        self.opt_budget = opt_budget
        self.model = RandomForestRegressor(random_state=self.seed, n_jobs=-1)  # use all processors

    def get_name(self):
        return "RF"

    def train(self, X, Y):
        assert(Y.shape[1] == 1)
        Y = Y.squeeze() if Y.shape[0] > 1 else Y
        if self.optimize:
            opt_space = [
                Integer(1, 1000, name="n_estimators"), 
                Integer(2, int(len(X)), name="min_samples_split"),
                Categorical(["sqrt", "log2", None], name="max_features"),
            ]
            @use_named_args(opt_space)
            def _opt_objective(**params):
                self.model.set_params(**params)
                return -np.mean(cross_val_score(self.model, X, Y, cv=5, n_jobs=-1, scoring="neg_mean_absolute_error"))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_gp = gp_minimize(_opt_objective, opt_space, n_calls=self.opt_budget, random_state=self.seed)
            self.optimal_parameters = res_gp.x
            print(f"Score: {res_gp.fun}")
            print(f"Parameters: N={res_gp.x[0]}, Split-fract={res_gp.x[1]}, max-feat={res_gp.x[2]}")
            self.model = RandomForestRegressor(
                        n_estimators=res_gp.x[0], 
                        min_samples_split=res_gp.x[1], 
                        max_features=res_gp.x[2],
                        random_state=self.seed, 
                        n_jobs=-1,
                        ) 
        self.model.fit(X, Y)

    def predict(self, X):
        pred = self.model.predict(X).reshape(-1, 1)
        unc = np.zeros(pred.shape)
        return pred, unc

    def predict_f(self, X):
        return self.predict(X)
