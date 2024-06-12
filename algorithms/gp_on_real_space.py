import warnings
from statistics import variance

import gpflow.optimizers
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import Parameter
from gpflow.ci_utils import ci_niter
from gpflow.kernels.linears import Linear
from gpflow.mean_functions import Constant, Zero
from gpflow.models import GPR
from gpflow.optimizers import Scipy
from gpflow.utilities import print_summary, set_trainable, to_default_float
from packaging.version import parse

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
        self.noise_variance = 0.2
        self.kernel_variance = 0.4
        self.init_length = init_length
        self.opt_success = False

    def get_name(self):
        return "GP" + self.kernel_factory().name

    def prior(self, X, Y):
        self.gp = GPR(
            data=(tf.constant(X, dtype=tf.float64), tf.constant(Y, dtype=tf.float64)),
            kernel=self.kernel_factory(),
            mean_function=self.mean_function,
            noise_variance=self.noise_variance,
        )
        if self.gp.kernel.__class__ != Linear:
            self.gp.kernel.lengthscales = Parameter(
                self.init_length,
                transform=tfp.bijectors.Softplus(),
                prior=tfp.distributions.InverseGamma(
                    to_default_float(3.0), to_default_float(3.0)
                ),
            )
        self.gp.kernel.variance = Parameter(
            self.kernel_variance,
            transform=tfp.bijectors.Softplus(),
            prior=tfp.distributions.InverseGamma(
                to_default_float(3.0), to_default_float(3.0)
            ),
        )
        self.gp.likelihood.variance = Parameter(
            value=self.noise_variance,
            transform=tfp.bijectors.Softplus(),
            prior=tfp.distributions.Uniform(
                to_default_float(0.01), to_default_float(1.0)
            ),
        )
        return self.gp

    def train(self, X, Y):
        assert Y.shape[1] == 1
        self.prior(X, Y)
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

            def eval_func(
                closure, variables, compile=True, allow_unused_variables=None
            ):
                def _tf_eval(x):
                    values = cls.unpack_tensors(variables, x)
                    cls.assign_tensors(variables, values)
                    if parse(scipy.__version__) >= parse("1.8.0"):
                        loss, grads = (
                            gpflow.optimizers.scipy._compute_loss_and_gradients(
                                closure,
                                variables,
                                unconnected_gradients=tf.UnconnectedGradients.ZERO,
                            )
                        )
                    else:
                        loss, grads = (
                            gpflow.optimizers.scipy._compute_loss_and_gradients(
                                closure, variables
                            )
                        )
                    return loss, cls.pack_tensors(grads)

                if compile:
                    _tf_eval = tf.function(_tf_eval)

                def _eval(x):
                    try:
                        loss, grad = _tf_eval(tf.convert_to_tensor(x))
                        loss = np.nan_to_num(
                            loss.numpy().astype(np.float64), nan=np.inf
                        )
                        grad = np.nan_to_num(
                            grad.numpy().astype(np.float64), nan=np.inf
                        )
                        return loss, grad
                    except TypeError as e:
                        warnings.warn(
                            "The optimizer tried a numerically unstable parameter configuration. Trying to recover optimization..."
                        )
                        return np.finfo(np.float64).max, np.ones(x.shape)  # go back
                    except tf.errors.InvalidArgumentError as e:
                        warnings.warn("Matrix Inversion failed! Exiting...")

                return _eval

            opt.eval_func = eval_func
            try:
                opt_logs = opt.minimize(
                    closure=self.gp.training_loss,
                    variables=self.gp.trainable_variables,
                    method="BFGS",
                    options=dict(maxiter=ci_niter(500)),
                )
                self.opt_success = True
            except TypeError as e:
                warnings.warn("Optimization failed! Exiting...")
            pass
