import warnings
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from algorithms.abstract_algorithm import AbstractAlgorithm
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.ensemble._base import BaseEnsemble, _partition_estimators
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn._config import config_context, get_config
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
import numpy as np
import threading
from joblib import Parallel
from functools import update_wrapper
import functools


# remove when https://github.com/joblib/joblib/issues/1071 is fixed
def delayed(function):
    """Decorator used to capture the arguments of a function."""
    @functools.wraps(function)
    def delayed_function(*args, **kwargs):
        return _FuncWrapper(function), args, kwargs
    return delayed_function


class _FuncWrapper:
    """Load the global configuration before calling the function."""
    def __init__(self, function):
        self.function = function
        self.config = get_config()
        update_wrapper(self, self.function)

    def __call__(self, *args, **kwargs):
        with config_context(**self.config):
            return self.function(*args, **kwargs)


def _accumulate_uncertain_prediction(predict, X, out1, out2, lock):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    with lock:
        if len(out1) == 1:
            out1[0] += prediction
            out2[0] += prediction**2
        else:
            for i in range(len(out1)):
                out1[i] += prediction[i]
                out2[i] += prediction[i]**2


class Uncertain_RandomForestRegressor(RandomForestRegressor):
    def predict(self, X):
        """
        Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self, attributes="estimators_")
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
            y_hat2 = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)
            y_hat2 = np.zeros((X.shape[0]), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_uncertain_prediction)(e.predict, X, [y_hat], [y_hat2], lock)
            for e in self.estimators_)

        # y_hat /= len(self.estimators_)
        # y_hat2 /= len(self.estimators_)
        assert len(y_hat) == len(self.estimators_)
        print(len(self.estimators_))
        y_hat = np.divide(y_hat, len(self.estimators_))
        y_hat2 = np.divide(y_hat2, len(self.estimators_))
        
        return y_hat, (y_hat2) - y_hat**2


class UncertainRandomForest(AbstractAlgorithm):
    def __init__(self, optimize=False, seed=42, opt_budget=20, cached=True):
        self.model = None
        self.optimize = optimize
        self.seed = seed
        self.opt_budget = opt_budget
        self._persist_id = None
        self._cached = cached

    def get_name(self):
        return "uncertainRF"

    def _optimize_regressor(self, X, Y) -> Uncertain_RandomForestRegressor:
        # NOTE: use RF Regressor for optimize, to avoid parallel processing inconsistencies from predict
        # NOTE: RF and UncertainRF are fundamentally the same, except for predict function, after training
        model = RandomForestRegressor(random_state=self.seed, n_jobs=-1)  # use all processors
        opt_space = [
            Integer(2, 3000, name="n_estimators"),
        ]
        @use_named_args(opt_space)
        def _opt_objective(**params):
            self.model.set_params(**params) # NOTE: Optimization cannot be multi-threaded 
            return -np.mean(cross_val_score(model, X, Y, cv=3, n_jobs=-1, scoring="neg_mean_absolute_error", error_score="raise")) # TODO: multithreading fails
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_gp = gp_minimize(_opt_objective, opt_space, n_calls=self.opt_budget, random_state=self.seed)
        self.optimal_parameters = res_gp.x
        print(f"Score: {res_gp.fun}")
        print(f"Parameters: N={res_gp.x[0]}")
        model = Uncertain_RandomForestRegressor(
                    n_estimators=res_gp.x[0],
                    random_state=self.seed, 
                    n_jobs=1,
                    )
        return model


    def train(self, X, Y):
        assert(Y.shape[1] == 1)
        #Y = Y.squeeze() if Y.shape[0] > 1 else Y
        if self.optimize:
            if self._cached and self._persist_id:
                # NOTE: use hash, data, splitter, split when persisting optimal parameters
                filename = f"./results/cache/RF_optimal_estimators_{self._persist_id}.pkl"
                with open(filename, "rb") as infile:
                    loaded_optimal_parameters = pickle.load(infile)
                self.model = Uncertain_RandomForestRegressor(
                    n_estimators=loaded_optimal_parameters.get("n_estimators"),
                    random_state=self.seed, 
                    n_jobs=1,
                    )
            else:
                self.model = self._optimize_regressor(self, X, Y)
        else:
            self.model = Uncertain_RandomForestRegressor(random_state=self.seed, n_jobs=1)
        self.model.fit(X, Y.ravel())

    def predict(self, X, return_var=False):
        out = self.model.predict(X)
        pred, unc = out[0], out[1]
        return pred.reshape(-1, 1), unc.reshape(-1, 1)

    def predict_f(self, X):
        return self.predict(X)
