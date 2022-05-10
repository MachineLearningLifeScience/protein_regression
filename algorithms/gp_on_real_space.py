from statistics import variance
import warnings
import scipy
from packaging.version import parse

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow.optimizers
from gpflow import Parameter
from gpflow.utilities import to_default_float, set_trainable
from gpflow.ci_utils import ci_niter
from gpflow.mean_functions import Constant, Zero
from gpflow.optimizers import Scipy
from gpflow.models import GPR
from gpflow.kernels.linears import Linear

from algorithms.abstract_algorithm import AbstractAlgorithm
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

class GPonRealSpace(AbstractAlgorithm):
    def __init__(self, kernel_factory=lambda: Linear(), optimize=True, init_length=0.1):
        self.gp = None
        self.kernel_factory = kernel_factory
        self.optimize = optimize
        self.mean_function = Zero()
        self.noise_variance = 0.1
        self.init_length = init_length
        self.opt_success = False

    def get_name(self):
        return "GP" + self.kernel_factory().name

    def train(self, X, Y):
        assert(Y.shape[1] == 1)
        self.gp = GPR(data=(tf.constant(X, dtype=tf.float64), tf.constant(Y, dtype=tf.float64)), kernel=self.kernel_factory(),
                      mean_function=self.mean_function, noise_variance=self.noise_variance)
        self.gp.kernel.variance.assign(0.4)
        if self.gp.kernel.__class__ == gpflow.kernels.SquaredExponential:
            self.gp.kernel.lengthscales = Parameter(self.init_length, transform=tfp.bijectors.Softplus(), 
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
                    if parse(scipy.__version__) >= parse("1.8.0"):
                        loss, grads = gpflow.optimizers.scipy._compute_loss_and_gradients(closure, variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                    else:
                        loss, grads = gpflow.optimizers.scipy._compute_loss_and_gradients(closure, variables)
                    return loss, cls.pack_tensors(grads)
                if compile:
                    _tf_eval = tf.function(_tf_eval)

                def _eval(x):
                    try:
                        loss, grad = _tf_eval(tf.convert_to_tensor(x))
                        return loss.numpy().astype(np.float64), grad.numpy().astype(np.float64)
                    except TypeError as e:
                        warnings.warn("The optimizer tried a numerically unstable parameter configuration. Trying to recover optimization...")
                        return np.finfo(np.float64).max, np.ones(x.shape)  # go back
                    except tf.errors.InvalidArgumentError as e: 
                        warnings.warn("Matrix Inversion failed! Exiting...")

                return _eval
            opt.eval_func = eval_func
            try:
                opt_logs = opt.minimize(closure=self.gp.training_loss, variables=self.gp.trainable_variables, method="BFGS", 
                                        options=dict(maxiter=ci_niter(500)))
                self.opt_success = True
            except TypeError as e:
                warnings.warn("Optimization failed! Exiting...")
            pass
