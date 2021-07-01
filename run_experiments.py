from gpflow.kernels import SquaredExponential, Linear, Polynomial

from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from data.load_dataset import get_wildtype, get_alphabet
from data.train_test_split import pos_per_fold_assigner, positional_splitter, BlockPostionSplitter
from run_single_regression_task import run_single_regression_task


dataset = "1FQG"
representation = "transformer"
alphabet = get_alphabet(dataset)
wt = get_wildtype(dataset)


run_single_regression_task(dataset=dataset, method=GPOneHotSequenceSpace(alphabet_size=len(alphabet), kernel=Linear()),
                           train_test_splitter=BlockPostionSplitter(dataset=dataset))
#run_single_regression_task(dataset=dataset, method=GPOneHotSequenceSpace(alphabet_size=len(alphabet), kernel=Polynomial(degree=7.) + Linear() + Linear() * Polynomial(degree=7.)), train_test_splitter=block_position_splitter)
#run_single_regression_task(dataset=dataset, representation=representation, method=GPonRealSpace(kernel=Polynomial(degree=7.)), train_test_splitter=block_position_splitter)
#run_single_regression_task(dataset=dataset, representation=representation,
#                           method=GPonRealSpace(kernel=Polynomial(degree=7.) + SquaredExponential() + SquaredExponential() * Polynomial(degree=7.)), train_test_splitter=BlockPostionSplitter(dataset=dataset))
