import numpy as np
from sklearn.ensemble import RandomForestRegressor
from algorithms.abstract_algorithm import AbstractAlgorithm
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.ensemble._base import BaseEnsemble, _partition_estimators
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn._config import config_context, get_config
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
    """"Load the global configuration before calling the function."""
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
        check_is_fitted(self)
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

        y_hat /= len(self.estimators_)
        y_hat2 /= len(self.estimators_)
        
        return y_hat, np.sqrt((y_hat2) - y_hat**2) 


class UncertainRandomForest(AbstractAlgorithm):
    def __init__(self):
        self.model = None

    def get_name(self):
        return "uncertainRF"

    def train(self, X, Y):
        assert(Y.shape[1] == 1)
        self.model = Uncertain_RandomForestRegressor(random_state=42, n_jobs=-1)  # use all processors
        self.model.fit(X, Y.ravel())

    def predict(self, X):
        out = self.model.predict(X)
        pred, unc = out[0], out[1]
        return pred.reshape(-1, 1), unc.reshape(-1, 1)
