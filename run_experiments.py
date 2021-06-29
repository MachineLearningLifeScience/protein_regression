from gpflow.kernels import SquaredExponential, Linear, Polynomial, Kernel
from gpflow import default_float

from algorithms.linear_gp import GPOneHotSequenceSpace
from data.load_dataset import get_wildtype, get_alphabet
from data.train_test_split import pos_per_fold_assigner, positional_splitter
from run_single_regression_task import run_single_regression_task


dataset = "1FQG"
alphabet = get_alphabet(dataset)
wt = get_wildtype(dataset)


import tensorflow as tf
class BrownianMotion(Kernel):
    def __init__(self):
        super().__init__(name="BrownianMotionSum")

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        ret = tf.zeros([X.shape[0], X2.shape[0]], dtype=default_float())
        for d in range(X.shape[1]):
            ret += tf.reduce_min(X[:, d:d+1] * tf.transpose(X2[:, d:d+1]), axis=1)
        return ret

    def K_diag(self, X):
        return tf.reduce_sum(X, axis=1)


def block_position_splitter(X):
    return positional_splitter(X, wt, val=True, offset=4, pos_per_fold=pos_per_fold_assigner(dataset))


#run_single_regression_task(dataset=dataset, method=GPOneHotSequenceSpace(alphabet_size=len(alphabet), kernel=Linear()), train_test_splitter=block_position_splitter)
run_single_regression_task(dataset=dataset, method=GPOneHotSequenceSpace(alphabet_size=len(alphabet), kernel=Polynomial(degree=7.) + Linear() + Linear() * Polynomial(degree=7.)), train_test_splitter=block_position_splitter)
