import tensorflow as tf
from gpflow.mean_functions import Constant
from gpflow.models import GPR
from gpflow.kernels.linears import Linear

from algorithms.gp_on_real_space import GPonRealSpace
from util.numpy_one_hot import numpy_one_hot_2dmat


class GPOneHotSequenceSpace(GPonRealSpace):
    def __init__(self, alphabet_size: int, kernel_factory=lambda: Linear(), optimize=True):
        super().__init__(kernel_factory=kernel_factory, optimize=optimize)
        self.alphabet_size = alphabet_size

    def train(self, X, Y):
        assert(Y.shape[1] == 1)
        self.gp = GPR(data=(tf.constant(X.astype(float)), tf.constant(Y.astype(float))), kernel=self.kernel_factory(),
                      mean_function=Constant(), noise_variance=1e-3)
        self._optimize()

    def predict(self, X):
        μ, var = self.gp.predict_y(tf.constant(X.astype(float)))
        return μ, var

    def predict_f(self, X):
        μ, var = self.gp.predict_f(tf.constant(X.astype(float)))
        return μ, var