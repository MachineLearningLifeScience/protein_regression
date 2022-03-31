import warnings

import numpy as np
import tensorflow as tf
import gpflow.optimizers
from gpflow.mean_functions import Constant, Zero
from gpflow.optimizers import Scipy
from gpflow.models import GPR
from gpflow.kernels.linears import Linear

from algorithms.abstract_algorithm import AbstractAlgorithm


class GPonRealSpace(AbstractAlgorithm):
    def __init__(self, kernel_factory=lambda: Linear(), optimize=True):
        self.gp = None
        self.kernel_factory = kernel_factory
        self.optimize = optimize
        self.initial_noise = 1e-3
        self.mean_function = Zero()

    def get_name(self):
        return "GP" + self.kernel_factory().name

    def train(self, X, Y):
        assert(Y.shape[1] == 1)
        self.gp = GPR(data=(tf.constant(X), tf.constant(Y)), kernel=self.kernel_factory(),
                      mean_function=self.mean_function, noise_variance=self.initial_noise)
        self._optimize()

    def predict(self, X):
        μ, var = self.gp.predict_y(tf.constant(X))
        return μ, var

    def predict_f(self, X):
        μ, var = self.gp.predict_f(tf.constant(X))
        return μ, var

    def _optimize(self):
        if self.optimize:
            opt = Scipy()

            cls = opt
            def eval_func(closure, variables, compile=True):
                def _tf_eval(x):
                    values = cls.unpack_tensors(variables, x)
                    cls.assign_tensors(variables, values)
                    loss, grads = gpflow.optimizers.scipy._compute_loss_and_gradients(closure, variables)
                    return loss, cls.pack_tensors(grads)

                if compile:
                    _tf_eval = tf.function(_tf_eval)

                def _eval(x):
                    try:
                        loss, grad = _tf_eval(tf.convert_to_tensor(x))
                        return loss.numpy().astype(np.float64), grad.numpy().astype(np.float64)
                    except Exception as e:
                        #return np.nan, np.nan * np.ones(x.shape)
                        warnings.warn("The optimizer tried a numerically unstable parameter configuration. Trying to recover optimization...")
                        return np.finfo(np.float64).max, np.ones(x.shape)  # go back

                return _eval
            opt.eval_func = eval_func
            opt_logs = opt.minimize(self.gp.training_loss, self.gp.trainable_variables, method="BFGS",
                                    options=dict(maxiter=100))
            pass
