from operator import invert
import pytest
import numpy as np
from data.train_test_split import positional_splitter

np.random.seed(1234)
N = 500
L = 200
AA = 19
random_X = np.random.randint(0, AA, [N, L])  # unrelated multivariates
ref_X = np.random.randint(0, AA, [L])
# simulate SSL experiments
ssl_X = []
for i in range(L):
    x = ref_X.copy()
    for j in range(AA):
        x[i] = j
        ssl_X.append(x)
ssl_X = np.vstack(ssl_X)


def test_return_indices_lists():
    """
    Basic Test
    Case: positional splitter is invoked with input matrix and reference sequence
    Then: train and test indices are returned
    """
    train, _, test = positional_splitter(
        ssl_X, ref_X, val=False, offset=1, pos_per_fold=1
    )
    assert isinstance(train, list) and isinstance(test, list)
    assert len(train) == len(test)


def test_training_data_not_in_testing():
    """
    Case: positional splitter is invoked and returns train and test list
    Then: elements from test are not in training and elements in training are not in test
    """
    train, _, test = positional_splitter(
        ssl_X, ref_X, val=False, offset=1, pos_per_fold=1
    )
    test_not_in_train = [bool(seq not in train) for seq in test]
    train_not_in_test = [bool(seq not in test) for seq in train]
    assert all(test_not_in_train) and all(train_not_in_test)


def test_pos_per_fold():
    """
    Case: number of train and test splits and pos per fold is one, two, ..., n
    Then: test split is then number of unique mutations
    """
    mutation_pos = [np.argmax(ref_X != s) for s in ssl_X]
    unique_mutations = list(set(mutation_pos))
    mutation_pos = np.vstack(mutation_pos)
    for pos in range(1, int(L / 10)):
        print(pos)
        train, _, test = positional_splitter(
            ssl_X, ref_X, val=False, offset=0, pos_per_fold=pos
        )
        assert len(train) == len(test) == np.ceil(len(unique_mutations) / pos)
        for idx, _test in enumerate(test):
            pos_idx = idx * pos
            mutated_positions_at_idx = np.hstack(
                [
                    np.where(mutation_pos == p)[0]
                    for p in list(unique_mutations[pos_idx : pos_idx + pos])
                ]
            )
            np.testing.assert_equal(mutated_positions_at_idx, _test)


def test_no_offset_no_sequences_excluded_():
    """
    Case: when matrix and sequence reference is provided with zero offset
    Then: no sequences are missing
    """
    train, _, test = positional_splitter(
        ssl_X, ref_X, val=False, offset=0, pos_per_fold=1
    )
    for _train, _test in zip(train, test):
        assert (len(_train) + len(_test)) == len(ssl_X)


def test_with_offset_sequences_are_excluded():
    """
    Case: matrix and sequence reference are provided with offset value,
        which should exclude these positions from training/testing by proximity to pos_per_fold
    Then: Testing is X% of training minus fraction from offset
    """
    for offset in range(10):
        train, _, test = positional_splitter(
            ssl_X, ref_X, val=False, offset=offset, pos_per_fold=1
        )
        assert (len(train[0]) + len(test[0])) == len(ssl_X) - (offset * AA)


def test_with_offset_no_overlaps():
    """
    Case: If offset and position is given
    Then: sequences should be excluded from training that are around position+offset range around mutation
    """
    pos = 10
    mutation_pos = [np.argmax(ref_X != s) for s in ssl_X]
    unique_mutations = list(set(mutation_pos))
    mutation_pos = np.vstack(mutation_pos)
    for offset in range(1, 10):
        # given a train/test split with offset and position per fold
        train, _, test = positional_splitter(
            seqs=ssl_X, query_seq=ref_X, val=False, offset=offset, pos_per_fold=pos
        )
        for idx, (_train, _test) in enumerate(zip(train, test)):
            # the offset range at a position ...
            for k in range(1, offset):
                not_contained_idx = int(pos * idx) + k
                mutated_positions_at_exclusion_idx = np.hstack(
                    [np.where(mutation_pos == unique_mutations[not_contained_idx])[0]]
                )
                # ... should not be contained in training - TODO: but contained in testing instead??
                assert all(np.isin(mutated_positions_at_exclusion_idx, _test))
                assert all(
                    np.isin(mutated_positions_at_exclusion_idx, _train, invert=True)
                )


def test_validation_flag_false():
    """
    Case: when validation false
    Then: return empty list of same length as train test
    """
    train, val, test = positional_splitter(
        ssl_X, ref_X, val=False, offset=1, pos_per_fold=20
    )
    assert len(train) == len(val) == len(test)
    assert all([[not v for v in _val] for _val in val])


def test_validation_flag_return():
    train, val, test = positional_splitter(
        ssl_X, ref_X, val=True, offset=1, pos_per_fold=10
    )
    assert isinstance(val, list)
    assert len(train) == len(val) == len(test)


def test_validation_not_in_train_test():
    """
    TODO
    """
    pass


def test_sequences_equal_raises_error():
    """
    Case: no mutation in sequence dataset
    Then: Value Error gets raised and splitter fails
    """
    stacked_reference = np.vstack([ref_X for _ in range(100)])
    with pytest.raises(ValueError):
        positional_splitter(
            seqs=stacked_reference, query_seq=ref_X, val=True, offset=1, pos_per_fold=1
        )


def test_too_few_mutations_raises_error():
    """
    Case: too few mutational sequences
    Then: Value Error raised and splitter fails
    """
    with pytest.raises(ValueError):
        positional_splitter(ssl_X[:3], ref_X, val=False, offset=1, pos_per_fold=1)


def test_too_many_positions_per_split_raises_error():
    """
    Case: too few mutational sequences
    Then: Value Error raised and splitter fails
    """
    with pytest.raises(ValueError):
        positional_splitter(ssl_X[:3], ref_X, val=False, offset=0, pos_per_fold=L - 2)
