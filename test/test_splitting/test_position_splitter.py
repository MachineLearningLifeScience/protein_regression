import numpy as np
from data.train_test_split import positional_splitter

N = 500
L = 200
X = np.random.randint(0, 19, [N, L])
ref_X = np.random.randint(0, 19, [1, L])


def test_return_indices():
    train, val, test = positional_splitter(X, ref_X, val=False, offset=2, pos_per_fold=5)
    assert np.any(train) and np.any(val) and np.any(test)


def test_offset():
    assert False


def test_pos_per_fold():
    assert False


def test_sequences_equal():
    assert False


def test_short_sequences():
    assert False


def test_few_mutations():
    assert False