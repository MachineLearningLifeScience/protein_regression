from gpflow.kernels import SquaredExponential, Linear, Polynomial

from algorithms.linear_gp import GPOneHotSequenceSpace
from data.load_dataset import get_wildtype, get_alphabet
from data.train_test_split import pos_per_fold_assigner, positional_splitter
from run_single_regression_task import run_single_regression_task


dataset = "1FQG"
alphabet = get_alphabet(dataset)
wt = get_wildtype(dataset)


def block_position_splitter(X):
    return positional_splitter(X, wt, val=True, offset=4, pos_per_fold=pos_per_fold_assigner(dataset))


run_single_regression_task(dataset=dataset, method=GPOneHotSequenceSpace(alphabet_size=len(alphabet), kernel=Linear()), train_test_splitter=block_position_splitter)
#run_single_regression_task(dataset=dataset, method=GPOneHotSequenceSpace(alphabet_size=len(alphabet), kernel=Polynomial(degree=7.) + Linear() + Linear() * Polynomial(degree=7.)), train_test_splitter=block_position_splitter)
