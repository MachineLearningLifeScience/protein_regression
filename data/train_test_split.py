from builtins import ValueError

import numpy as np
from sklearn.model_selection import KFold

from data.load_dataset import get_wildtype, load_dataset


class AbstractTrainTestSplitter:
    def __init__(self):
        self.name = type(self).__name__

    def split(self, X):
        raise NotImplementedError()

    def get_name(self):
        return self.name


class RandomSplitter(AbstractTrainTestSplitter):
    def __init__(self, seed: int = 42):
        super().__init__()
        self.seed = seed

    def split(self, X):
        kf = KFold(n_splits=10, random_state=self.seed, shuffle=True)
        train_indices = []
        test_indices = []
        for train, test in kf.split(X):
            train_indices.append(train)
            test_indices.append(test)
        return train_indices, None, test_indices


class BioSplitter(AbstractTrainTestSplitter):
    def __init__(self, dataset, n_mutations: int=3, fold: int=3, shuffle=True):
        super().__init__()
        self.wt = get_wildtype(dataset)
        self.n_mutations = n_mutations
        self.fold = fold
        self.shuffle = shuffle
    
    def split(self, X):
        data_fraction = X.shape[0] / self.fold
        diff_to_wt = np.sum(self.wt != X, axis=1)
        X_below_threshold = np.random.shuffle(np.where(diff_to_wt <= self.n_mutations)[0])
        X_above_threshold = np.random.shuffle(np.where(diff_to_wt > self.n_mutations)[0])
        train_indices = []
        test_indices = []
        for idx in np.arange(0, X.shape[0], data_fraction):
            train = X_below_threshold[idx:idx+data_fraction]
            test = X_above_threshold[idx:idx+data_fraction]
            train_indices.append(train)
            test_indices.append(test)
        return train_indices, None, test_indices



class BlockPostionSplitter(AbstractTrainTestSplitter):
    def __init__(self, dataset):
        super().__init__()
        self.wt = get_wildtype(dataset)
        self.pos_per_fold = pos_per_fold_assigner(dataset)

    def split(self, X):
        return positional_splitter(X, self.wt, val=False, offset=4, pos_per_fold=self.pos_per_fold)


def positional_splitter(seqs, query_seq, val, offset, pos_per_fold):
    # offset is the positions that will be dropped between train and test positions 
    # to not allow info leakage just because positions are neighbours
    # Split_by_DI implements that positions are also split by direct info based on a "threshold"
    # needs an aln_path to work (fasta file)
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


def pos_per_fold_assigner(name: str):
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
