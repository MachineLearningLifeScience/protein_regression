import tensorflow as tf
from gpflow.mean_functions import Constant
from gpflow.models import GPR
from gpflow.kernels.linears import Linear

from algorithms.gp_on_real_space import GPonRealSpace
from util.numpy_one_hot import numpy_one_hot_2dmat


class GPOneHotSequenceSpace(GPonRealSpace):
    def __init__(self, alphabet_size: int, kernel=Linear(), optimize=True):
        super().__init__(kernel=kernel, optimize=optimize)
        self.alphabet_size = alphabet_size

    def train(self, X, Y):
        assert(Y.shape[1] == 1)
        self.gp = GPR(data=(tf.constant(X), tf.constant(Y)),
                      kernel=self.kernel, mean_function=Constant())
        self._optimize()

    def predict(self, X):
        return self.gp.predict_f(tf.constant(X))
