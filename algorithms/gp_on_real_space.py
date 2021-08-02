import numpy as np
import tensorflow as tf
from gpflow.mean_functions import Constant
from gpflow.optimizers import Scipy
from gpflow.models import GPR
from gpflow.kernels.linears import Linear

from algorithms.abstract_algorithm import AbstractAlgorithm


class GPonRealSpace(AbstractAlgorithm):
    def __init__(self, kernel=Linear(), optimize=True):
        self.gp = None
        self.kernel = kernel
        self.optimize = optimize

    def get_name(self):
        return "GP" + self.kernel.name

    def train(self, X, Y):
        assert(Y.shape[1] == 1)
        self.gp = GPR(data=(tf.constant(X), tf.constant(Y)),
                      kernel=self.kernel, mean_function=Constant())
        self._optimize()

    def predict(self, X):
        return self.gp.predict_f(tf.constant(X))

    def _optimize(self):
        if self.optimize:
            opt = Scipy()
            def callable(*args, **kwargs):
                try:
                    return self.gp.training_loss(*args, **kwargs)
                except Exception as e:
                    return tf.constant(np.NAN)

            opt_logs = opt.minimize(callable, self.gp.trainable_variables, method="BFGS",
                                    options=dict(maxiter=100))
