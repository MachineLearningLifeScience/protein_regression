import pytest
import numpy as np
from data import load_dataset
from data import RandomSplitter
from data import PositionSplitter
from data import FractionalRandomSplitter
from data import BioSplitter


@pytest.mark.parametrize("splitter", [RandomSplitter, PositionSplitter, BioSplitter])
def test_train_test_split_exclusive(splitter: callable):
    X, _ = load_dataset("1FQG")
    train, _, test = splitter("1FQG").split(X)
    for _train, _test in zip(train, test):
        assert all([t not in _test for t in _train])


@pytest.mark.parametrize("mutations", [(2, 2), (3, 3)])
def test_biosplitter_intradomain(mutations):
    X, _ = load_dataset("TOXI")
    train, _, test = BioSplitter("TOXI", mutations[0], mutations[1]).split(X)
    for _train, _test in zip(train, test):
        assert all([t not in _test for t in _train])
