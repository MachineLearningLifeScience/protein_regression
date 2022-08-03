"""
Reference CV splitter instances, for usage across all experimental runs and plotting
"""
from typing import List
import numpy as np
from typing import Callable
from algorithm_factories import get_key_for_factory
from data.train_test_split import AbstractTrainTestSplitter
from data.train_test_split import RandomSplitter, BlockPostionSplitter 
from data.train_test_split import PositionSplitter, BioSplitter
from data.train_test_split import FractionalRandomSplitter, WeightedTaskSplitter

# SPLITTING: random, block, positional, fractional-random, mutation-lvl


def BioSplitterFactory(dataset: str, n_mutations_train: int, n_mutations_test: int) -> List[AbstractTrainTestSplitter]:
    return [BioSplitter(dataset, n_mutations_train=n_mutations_train, n_mutations_test=n_mutations_test)]


def FractionalSplitterFactory(dataset: str, fractions: np.ndarray=None) -> List[AbstractTrainTestSplitter]:
    if not fractions:
        fractions = np.concatenate([np.arange(0.001, .3, 0.01), np.arange(.3, .6, 0.03), np.arange(.6, 1.05, 0.05)])
    return [FractionalRandomSplitter(dataset, fraction) for fraction in fractions]


def PositionalSplitterFactory(dataset: str, positions: int=15) -> List[AbstractTrainTestSplitter]:
    return [PositionSplitter(dataset, positions=positions)]


def BlockSplitterFactory(dataset) -> List[AbstractTrainTestSplitter]:
    return [BlockPostionSplitter(dataset)]


def RandomSplitterFactory(dataset) -> List[AbstractTrainTestSplitter]:
    return [RandomSplitter(dataset)]


def WeightedTaskSplitterFactory(dataset, threshold=3) -> List[AbstractTrainTestSplitter]:
    return [WeightedTaskSplitter(dataset, threshold=threshold)]


PROTOCOL_REGISTRY = {
    get_key_for_factory(f): f for f in [RandomSplitterFactory, BlockSplitterFactory, PositionalSplitterFactory, FractionalSplitterFactory, BioSplitterFactory]
}