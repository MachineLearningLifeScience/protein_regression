import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
from gpflow import Parameter
from gpflow.mean_functions import Constant
from gpflow.models import GPR
from gpflow.kernels.linears import Linear
from gpflow.utilities import to_default_float, set_trainable

from algorithms.gp_on_real_space import GPonRealSpace
from util.numpy_one_hot import numpy_one_hot_2dmat


class GPOneHotSequenceSpace(GPonRealSpace):
    def __init__(self, alphabet_size: int, kernel_factory=lambda: Linear(), optimize=True, init_length=0.1):
        super().__init__(kernel_factory=kernel_factory, optimize=optimize, init_length=init_length)
        self.alphabet_size = alphabet_size

    def train(self, X, Y):
        assert(Y.shape[1] == 1)
        self.gp = GPR(data=(tf.constant(X.astype(float)), tf.constant(Y.astype(float))), kernel=self.kernel_factory(),
                      mean_function=self.mean_function, noise_variance=self.noise_variance)
        self.gp.kernel.variance.assign(0.4)
        if self.gp.kernel.__class__ == gpflow.kernels.SquaredExponential:
            self.gp.kernel.lengthscales = Parameter(self.init_length, transform=tfp.bijectors.Softplus(), 
                                prior=tfp.distributions.Uniform(to_default_float(0.001), to_default_float(2)))
        self.gp.likelihood.variance = Parameter(value=self.noise_variance, transform=tfp.bijectors.Softplus(), 
                                prior=tfp.distributions.Uniform(to_default_float(0.01), to_default_float(0.2)))
        self._optimize()

    def predict(self, X):
        μ, var = self.gp.predict_y(tf.constant(X.astype(float)))
        return μ, var

    def predict_f(self, X):
        μ, var = self.gp.predict_f(tf.constant(X.astype(float)))
        return μ, var