from statistics import variance
import warnings

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow.optimizers
from gpflow import Parameter
from gpflow.utilities import to_default_float, set_trainable
from gpflow.mean_functions import Constant, Zero
from gpflow.optimizers import Scipy
from gpflow.models import GPR
from gpflow.kernels.linears import Linear
import seaborn as sns
import matplotlib.pyplot as plt

from algorithms.abstract_algorithm import AbstractAlgorithm
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

class GPonRealSpace(AbstractAlgorithm):
    def __init__(self, kernel_factory=lambda: Linear(), optimize=True, ard=False):
        self.gp = None
        self.kernel_factory = kernel_factory
        self.optimize = optimize
        self.mean_function = Zero()
        self.noise_variance = 0.1 #1e-3
        self.ard = ard

    def get_name(self):
        return "GP" + self.kernel_factory().name

    def train(self, X, Y):
        assert(Y.shape[1] == 1)
        self.gp = GPR(data=(tf.constant(X), tf.constant(Y)), kernel=self.kernel_factory(),
                      mean_function=self.mean_function, noise_variance=self.noise_variance)
        self.gp.kernel.variance.assign(0.4)
        if self.gp.kernel.__class__ == gpflow.kernels.SquaredExponential:
            self.gp.kernel.lengthscales = Parameter(0.1, transform=tfp.bijectors.Softplus(), 
                                prior=tfp.distributions.Uniform(to_default_float(0.001), to_default_float(2)))
        self.gp.likelihood.variance = Parameter(value=self.noise_variance, transform=tfp.bijectors.Softplus(), 
                                prior=tfp.distributions.Uniform(to_default_float(0.01), to_default_float(0.2)))
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
                                    options=dict(maxiter=500))
            pass
