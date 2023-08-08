import pytest
import numpy as np
from data import load_dataset
from data.load_dataset import load_one_hot
from data.load_dataset import get_wildtype_and_offset
from data.load_dataset import get_mutation_diff
from data.train_test_split import BioSplitter

toxi_sequences, _ = load_one_hot("TOXI")
toxi_wt, _ = get_wildtype_and_offset("TOXI")
toxi_data, toxi_obs = load_dataset("TOXI")

N_toxi_sequences = toxi_data.shape[0]

biosplitter = BioSplitter("TOXI")


def test_mutation_diff():
    x = np.array([1,2,3,4,5])
    y = np.array([1,3,3,4,5])
    diff = get_mutation_diff(x, y)
    assert diff == 1


@pytest.mark.parametrize("n_mutations", [1, 2, 3, 4])
def test_n_mutations_in_domain(n_mutations):
    bs = BioSplitter("TOXI", n_mutations_train=n_mutations, n_mutations_test=n_mutations)
    train, _, test = bs.split(toxi_data)
    for train_idx, test_idx in zip(train, test):
        diff_to_wt_train = np.array([get_mutation_diff(toxi_wt, seq) for seq in toxi_sequences[train_idx]])
        diff_to_wt_test = np.array([get_mutation_diff(toxi_wt, seq) for seq in toxi_sequences[test_idx]])
        assert all(diff_to_wt_train <= n_mutations)
        assert all(diff_to_wt_test <= n_mutations)


@pytest.mark.parametrize("n_mutations", [(1,2), (2,3), (3,4)])
def test_n_mutations_out_of_domain(n_mutations):
    bs = BioSplitter("TOXI", n_mutations_train=n_mutations[0], n_mutations_test=n_mutations[1])
    train, _, test = bs.split(toxi_data)
    for train_idx, test_idx in zip(train, test):
        diff_to_wt_train = np.array([get_mutation_diff(toxi_wt, seq) for seq in toxi_sequences[train_idx]])
        diff_to_wt_test = np.array([get_mutation_diff(toxi_wt, seq) for seq in toxi_sequences[test_idx]])
        assert all(diff_to_wt_train <= n_mutations[0])
        assert all(diff_to_wt_test == n_mutations[1])


def test_are_all_previous_mutations_contained_in_next_domains():
    sequences_train = []
    sequences_test = []
    for domain in range(1,3):
        bs = BioSplitter("TOXI", n_mutations_train=domain, n_mutations_test=domain+1)
        train, _, test = bs.split(toxi_data)
        sequences_train.append(train)
        sequences_test.append(test)
    for idx in range(len(sequences_train)-1):
        assert all(np.in1d(sequences_train[idx], sequences_train[idx+1]))
        assert all(np.in1d(sequences_test[idx], sequences_train[idx+1])) # Test of D_n should be in train of D_n+1

    # TODO: test that all singles are in next domain and doubles are in next etc.


@pytest.mark.parametrize("n_mutations", [1, 2, 3, 4])
def test_setsize_in_domain_equal_size(n_mutations):
    bs = BioSplitter("TOXI", n_mutations_train=n_mutations, n_mutations_test=n_mutations)
    train, _, test = bs.split(toxi_data)
    target_val = len(train[0]) + len(test[0])
    np.testing.assert_array_equal(np.array([len(train_idx)+len(test_idx) for train_idx, test_idx in zip(train, test)]), target_val)


@pytest.mark.parametrize("n_mutations", [(1,1), (1,2), (2,2), (2,3), (3,3), (3,4), (4,4)])
def test_set_indices_unique(n_mutations):
    bs = BioSplitter("TOXI", n_mutations_train=n_mutations[0], n_mutations_test=n_mutations[1])
    train, _, test = bs.split(toxi_data)
    for idx_train, idx_test in zip(train, test):
        np.testing.assert_array_equal(np.setdiff1d(idx_train, idx_test), idx_train)