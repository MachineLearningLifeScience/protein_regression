from builtins import ValueError
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from data.load_dataset import get_wildtype_and_offset, load_dataset, load_sequences_of_representation
from util.mlflow.constants import ONE_HOT


class AbstractTrainTestSplitter:
    def __init__(self):
        self.name = type(self).__name__

    def split(self, X):
        raise NotImplementedError()

    def get_name(self):
        return self.name


class RandomSplitter(AbstractTrainTestSplitter):
    def __init__(self, dataset, seed: int = 42):
        super().__init__()
        self.dataset = dataset
        self.seed = seed

    def split(self, X):
        kf = KFold(n_splits=10, random_state=self.seed, shuffle=True)
        train_indices = []
        test_indices = []
        for train, test in kf.split(X):
            train_indices.append(train)
            test_indices.append(test)
        return train_indices, None, test_indices


class FractionalRandomSplitter(AbstractTrainTestSplitter):
    """
    n-fold Crossvalidation with a pct. sub-selection for training.
    """
    def __init__(self, dataset:str, fraction: float, seed: int = 42, n_splits=5):
        super().__init__()
        self.seed = seed
        self.dataset = dataset
        self.fraction = fraction
        self.n_splits = n_splits

    def split(self, X):
        N = X.shape[0]
        n_sequences = np.ceil((N-N*(1/self.n_splits)) * self.fraction)
        train_indices = []
        test_indices = []
        kf = KFold(n_splits=self.n_splits, random_state=self.seed, shuffle=True)
        for train, test in kf.split(X):
            train_indices.append(train[:int(n_sequences)])
            test_indices.append(test[:int(n_sequences)])
        return train_indices, None, test_indices
    
    def get_name(self):
        splitter_name = f"{self.name}_{str(np.round(self.fraction, 3))}"
        return splitter_name


class BioSplitter(AbstractTrainTestSplitter):
    """
    Splitting Protocol that splits by amount of mutations compared to reference sequence.
    From n_mutations_train to n_mutations_test.
    """
    def __init__(self, dataset, n_mutations_train: int=3, n_mutations_test: int=4, test_fraction: float=0.2):
        super().__init__()
        self.dataset = dataset
        self.wt, _ = get_wildtype_and_offset(dataset)
        self.n_mutations_train = n_mutations_train
        self.n_mutations_test = n_mutations_test
        self.test_fraction = test_fraction
    
    def split(self, X, representation=ONE_HOT, missed_indices=None):
        """
        Splits input data by mutational threshold, such that below threshold is training
        and above threshold is testing.
        If mutations equal, train on all available data (below equal threshold) and test on subset of threshold.
        w.r.t. CV protocol either setup is effectively one split.
        """
        _X = load_sequences_of_representation(self.dataset, representation=representation)
        if missed_indices:
            _X = np.delete(_X, missed_indices, axis=0) 
        assert _X.shape[0] == X.shape[0]
        diff_to_wt = np.sum(self.wt != _X, axis=1)
        if self.n_mutations_train == self.n_mutations_test:
            all_indices = np.where(diff_to_wt <= self.n_mutations_train)[0]
            n_mutants_indices = np.where(diff_to_wt <= self.n_mutations_train)[0]
            np.random.shuffle(n_mutants_indices)
            test_indices = n_mutants_indices[-int(len(all_indices)*self.test_fraction):]
            train_indices = np.setdiff1d(all_indices, test_indices)[np.newaxis, :]
            test_indices = test_indices[np.newaxis, :]
        else:
            train_indices = np.where(diff_to_wt <= self.n_mutations_train)[0][np.newaxis, :]
            test_indices = np.where(diff_to_wt == self.n_mutations_test)[0][np.newaxis, :]
        return train_indices, None, test_indices
    
    def get_name(self):
        splitter_name = self.name+str(self.n_mutations_train)+"_"+str(self.n_mutations_test)
        return splitter_name


class PositionSplitter(AbstractTrainTestSplitter):
    """
    Splitting Protocol that splits by given positions range.
    """
    def __init__(self, dataset: str, positions: int=15, missing_indices: np.ndarray=None):
        super().__init__()
        self.wt, _ = get_wildtype_and_offset(dataset)
        self.dataset = dataset
        self.positions = positions

    def split(self, X, representation=ONE_HOT, missed_indices=None):
        """
        Splitting routine of Positionsplitter.
        Input: X - representation matrix
        Output: train, test indices
        Positionsplitter dependant on sequence encoding. This may mismatch with loaded X!
        Requires internal filtering.
        Either missing_idx exists, enforcing subselection with boolean mask.
        """
        _X = load_sequences_of_representation(self.dataset, representation)
        if missed_indices is not None:
            _X = np.delete(_X, missed_indices, axis=0) 
        assert _X.shape[0] == X.shape[0]
        return positional_splitter(_X, self.wt, val=False, offset=4, pos_per_fold=self.positions)
    
    def get_name(self):
        splitter_name = f"{self.name}_p{self.positions}"
        return splitter_name


class BlockPostionSplitter(AbstractTrainTestSplitter):
    def __init__(self, dataset):
        super().__init__()
        self.wt, _ = get_wildtype_and_offset(dataset)
        self.pos_per_fold = pos_per_fold_assigner(dataset)

    def split(self, X): # TODO broken for TOXI
        return positional_splitter(X, self.wt, val=False, offset=4, pos_per_fold=self.pos_per_fold)


def positional_splitter(seqs, query_seq, val, offset, pos_per_fold):
    # offset is the positions that will be dropped between train and test positions 
    # to not allow info leakage just because positions are neighbours
    # Split_by_DI implements that positions are also split by direct info based on a "threshold"
    # needs an aln_path to work (fasta file)
    # TODO: requires tests
    mut_pos = []
    for seq in seqs:
        mut_pos.append(np.argmax(query_seq != seq))
    unique_mut = np.unique(mut_pos)

    train_indices = []
    test_indices = []
    val_indices = []

    counter = 0
    for i in range(len(np.unique(mut_pos))//(pos_per_fold)+1):
        
        test_mut = list(unique_mut[counter:counter+pos_per_fold])
        if len(test_mut)==0:
            continue
        
        train_mut = list(unique_mut[:max(counter-offset, 0)]) +\
                    list(unique_mut[counter+pos_per_fold+offset:])
        
        if offset > 0:
            buffer_mut = list(unique_mut[max(counter-offset, 0):counter]) +\
                     list(unique_mut[counter+pos_per_fold:counter+pos_per_fold+offset])
        else:
            buffer_mut = []

        if val:
            val_mut = [unique_mut[-int(1/3*pos_per_fold):],
                      np.hstack([unique_mut[:int(1/6*pos_per_fold)], unique_mut[-int(1/6*pos_per_fold):]]),
                      unique_mut[:int(1/3*pos_per_fold)]]
            train_mut = [mut for mut in train_mut if mut not in val_mut[i]]
        else:
            val_mut = [[] for i in range(len(np.unique(mut_pos)))]

        test_idx = np.hstack([np.where(mut_pos==pos)[0] for pos in test_mut])
        train_idx = np.hstack([np.where(mut_pos==pos)[0] for pos in train_mut])

        if offset>0:
            buffer_idx = np.hstack([np.where(mut_pos==pos)[0] for pos in buffer_mut])
        else:
            buffer_idx = []

        if val:
            val_idx = np.hstack([np.where(mut_pos==pos)[0] for pos in val_mut[i]])
        else:
            val_idx = [[] for i in range(len(np.unique(mut_pos)))]
        
        verify_num_mut = len(test_mut) + len(train_mut) + len(buffer_mut) + len(val_mut[i])
        verify_num_idx = (len(test_idx) + len(train_idx) + len(buffer_idx)) + len(val_idx)
        assert len(list(set(test_mut).intersection(set(train_mut))))==0, "test and train idx overlap"
        assert len(list(set(train_idx).intersection(set(test_idx))))==0, "test and train idx overlap"
        assert len(unique_mut) == verify_num_mut, 'Something wrong with number of positions/mutations. Number of idx: '+\
                                                    str(verify_num_idx) + 'Number of mut:' + str(verify_num_mut)
        
        train_indices.append(train_idx)
        val_indices.append(val_idx)
        test_indices.append(test_idx)

        counter += pos_per_fold
        
    return train_indices, val_indices, test_indices


def pos_per_fold_assigner(name: str) -> int:
    """
    Return reference number of positions for blocksplitter.
    """
    if name.lower() == 'blat' or name == '1FQG':
        pos_per_fold = 85
    elif name.lower() == 'ubqt':
        pos_per_fold = 25
    elif name.lower() == 'brca':
        pos_per_fold = 63
    elif name.lower() == 'timb':
        pos_per_fold = 28
    elif name.lower() == 'calm':
        pos_per_fold = 47
    elif name.lower() == 'mth3':
        pos_per_fold = 107
    elif name.lower() == 'toxi':
        pos_per_fold = 31
    else:
        raise ValueError("Unknown dataset: %s" % name)
    return pos_per_fold
