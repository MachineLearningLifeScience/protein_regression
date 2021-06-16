import tensorflow as tf
from gpflow.optimizers import Scipy
from gpflow.models import GPR
from gpflow.kernels import Kernel
from gpflow.kernels.linears import Linear

from algorithms.abstract_algorithm import AbstractAlgorithm
from util.numpy_one_hot import numpy_one_hot_2dmat


class GPOneHotSequenceSpace(AbstractAlgorithm):
    def __init__(self, alphabet_size: int, kernel=Linear()):
        self.gp = None
        self.alphabet_size = alphabet_size
        self.kernel = kernel

    def get_name(self):
        return "GP" + self.kernel.name

    def train(self, X, Y):
        assert(Y.shape[1] == 1)
        self.gp = GPR(data=(tf.constant(numpy_one_hot_2dmat(X, max=self.alphabet_size)), tf.constant(Y)),
                      kernel=self.kernel)
        opt = Scipy()
        opt_logs = opt.minimize(self.gp.training_loss, self.gp.trainable_variables, options=dict(maxiter=1000))

    def predict(self, X):
        return self.gp.predict_f(tf.constant(numpy_one_hot_2dmat(X, max=self.alphabet_size)))
