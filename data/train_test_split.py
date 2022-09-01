from builtins import ValueError
from typing import Tuple
import numpy as np
from itertools import combinations
from numba import jit, float64
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
from data.load_dataset import get_wildtype_and_offset, load_dataset, load_sequences_of_representation
from util.log import prep_for_mutation
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
    def __init__(self, dataset, n_mutations_train: int=3, n_mutations_test: int=4, n_splits=5):
        super().__init__()
        self.dataset = dataset
        self.wt, _ = get_wildtype_and_offset(dataset)
        self.n_mutations_train = n_mutations_train
        self.n_mutations_test = n_mutations_test
        self.n_splits = n_splits
    
    def split(self, X, representation=ONE_HOT, missed_indices=None):
        """
        Splits input data by mutational threshold, such that below threshold is training
        and above threshold is testing.
        If mutations equal, train on all available data (below equal threshold) and test on subset of threshold.
        w.r.t. CV protocol either setup is effectively one split.
        """
        _X = load_sequences_of_representation(self.dataset, representation=representation)
        if missed_indices and len(X) != len(_X):
            _X = np.delete(_X, missed_indices, axis=0) 
        assert _X.shape[0] == X.shape[0]
        diff_to_wt = np.sum(self.wt != _X, axis=1)
        if self.n_mutations_train == self.n_mutations_test:
            train_indices, test_indices = [], []
            all_indices = np.where(diff_to_wt <= self.n_mutations_train)[0]
            n_mutants_indices = np.where(diff_to_wt == self.n_mutations_test)[0]
            N_test = len(n_mutants_indices) // self.n_splits
            np.random.shuffle(n_mutants_indices) # random permutation of target mutations
            for idx in range(self.n_splits):
                test_indices.append(n_mutants_indices[N_test*idx:N_test*idx+N_test])
                train_indices.append(np.setdiff1d(all_indices, test_indices))
        else: # No CV splitting possible for change in domains, case is: from all available k-M to k+1M
            train_indices = np.where(diff_to_wt <= self.n_mutations_train)[0][np.newaxis, :]
            test_indices = np.where(diff_to_wt == self.n_mutations_test)[0][np.newaxis, :]
        return train_indices, None, test_indices
    
    def get_name(self):
        splitter_name = self.name+str(self.n_mutations_train)+"_"+str(self.n_mutations_test)
        return splitter_name


class WeightedTaskSplitter(AbstractTrainTestSplitter):
    def __init__(self, dataset: str, seed=42, n_splits=5, threshold=2, X_p_fraction=0.05, p_accept=0.95, limit_N_Xs=250, split_type="mutation"):
        """
        Initialized with X_s, Y_s data distribution and n_splits for k-fold CV.

        INITIAL CONDITIONS if eta=0.75 to satisfy N-sum constraint, initial approx. alphas=1.5

        """
        super().__init__()
        self.dataset = dataset
        self.n_splits = n_splits
        self.seed = seed
        self.X_s, self.Y_s = None, None
        self.X_s_N_limit = limit_N_Xs # hard limit, due to optimization scaling inefficiencies 
        self.eta = 0.75
        if split_type not in ["mutation", "threshold"]:
            raise ValueError("Incorrect split_type provided")
        self.threshold = threshold
        self.split_type = split_type
        self.X_p_fraction = X_p_fraction
        self.p_accept = p_accept

    def split(self, X, y):
        """
        Implements indirect weighted inductive transfer learning regression (KL-ITL) 
        by weighting one distribution against another, given that a small subset of defined other distribution is given.
        Input X, y is split up into X_p, y_p and X_s, y_s
        """
        train_idx_list, test_idx_list = [], []
        # eta = self._eta_optimize(X, y)
        # return indices based on sampling from computed weights
        for _ in range(self.n_splits):
            if self.split_type == "mutation":
                assert type(self.threshold) == int
                p_idx, s_idx, test_idx = self.split_Xy_by_mutations(X)
            elif self.split_type == "threshold":
                assert type(self.threshold) == float
                p_idx, s_idx, test_idx = self.split_Xy_by_observations(X, y)
            X_p, y_p = X[p_idx], y[p_idx]
            s_idx = s_idx[:self.X_s_N_limit] # limit amount of X_s for later optimization
            self.X_s, self.Y_s = X[s_idx], y[s_idx]
            training_N = X_p.shape[0] + self.X_s.shape[0]
            optimal_alphas = self._optimize(X_p, y_p)
            weights = self.weight(alphas=optimal_alphas, x_p=X_p, y_p=y_p, X_s=self.X_s, Y_s=self.Y_s, eta=self.eta)
            avrg_weights = np.mean(weights, axis=0)
            # we have a weight for each X_py_p entry -> take average shape: [N_p, M_s], take average
            assert avrg_weights.shape[0] == self.X_s.shape[0]
            # training data: ALL X_p equal likelihood, all X_s data with avrg weights
            idx_vec = np.concatenate([p_idx, s_idx])
            training_N = len(p_idx)+avrg_weights.sum()
            p_vec = np.concatenate([np.ones(len(p_idx)), avrg_weights]) / training_N # normalize to make it p-measure
            assert idx_vec.shape == p_vec.shape
            train_idx = np.random.choice(idx_vec, size=int(np.ceil(training_N)), p=p_vec)
            train_idx_list.append(train_idx)
            test_idx_list.append(test_idx)
        return train_idx_list, None, test_idx_list

    def split_Xy_by_mutations(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split by number of mutations against wild-type: e.g. X_s = singles+doubles, X_p = subset(triples)
        Note: data-size condition |X_p| << |X_s|
        Returns: p, s, and test indices for X
        """
        wt, _ = get_wildtype_and_offset(self.dataset)
        _X = load_sequences_of_representation(self.dataset, representation=ONE_HOT)
        assert X.shape[0] == _X.shape[0]
        diff_to_wt = np.sum(wt != _X, axis=1)
        s_idx = np.where(diff_to_wt < self.threshold)[0]
        all_p_idx = np.where(diff_to_wt == self.threshold)[0]
        N_X_p = int(len(all_p_idx)*self.X_p_fraction)
        np.random.shuffle(all_p_idx)
        p_idx = all_p_idx[:N_X_p]
        test_idx = all_p_idx[N_X_p:]
        assert p_idx.shape[0] < s_idx.shape[0]
        assert p_idx.shape[0] + test_idx.shape[0] == all_p_idx.shape[0]
        return p_idx, s_idx, test_idx

    def split_Xy_by_observations(self, y) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split by functional threshold with acceptance probability.
        """
        s_idx, p_idx, test_idx = [], [], []
        all_indices = np.arange(0, len(y))
        p_idx = np.array([idx for idx, _y in enumerate(y) if _y >= self.threshold and bool(np.random.binomial(1, self.p_accept))]) # functional variants
        s_idx = np.setdiff1d(all_indices, s_idx) # non-functional variants
        np.random.shuffle(p_idx)
        N_X_p = int(len(p_idx)*self.X_p_fraction)
        test_idx = p_idx[:N_X_p]
        p_idx = p_idx[N_X_p:]
        assert len(p_idx) < len(s_idx) and len(p_idx) < len(test_idx)
        return p_idx, s_idx, test_idx

    def _eta_optimize(self, X, y, k=10):
        raise NotImplementedError("eta optimization not yet functional!")
        etas = np.linspace(0, 1, k)
        performance = []
        for eta in etas:
            alphas = self._optimize(X, y, eta)
            performance.append(np.sum(self.average_log_weight(alphas, X, y)))
        return np.max(etas)

    def constrained_weight_sum(self, alphas):
        """
        Constraint: N=sum from j=1 to N: w(X_s[j], Y_s[j])
        rewritten as 0==sum-N equality constraint
        """
        N = self.X_s.shape[0]
        w = self.weight(alphas, self.X_s, self.Y_s, self.X_s, self.Y_s, self.eta) # returns NxN matrix from euclidean distances
        # TODO: inquire if average across each sequence is sensible
        w = np.mean(w, axis=1)
        return np.sum(w) - N

    def _optimize(self, X, y):
        """
        Compute weights by optimizing on training split
        Returns: optimal alphas vector
        """
        N = self.X_s.shape[0]
        initial_alpha = 1.5
        alphas0 = np.ones(N)[:, np.newaxis] * initial_alpha
        cons = [{"type": "eq", 
                "fun": self.constrained_weight_sum,},
                {"type": "ineq",
                "fun": lambda alpha: alpha,}]
                # "jac": lambda alpha: np.ones(alpha.shape[0])[np.newaxis, :]}]
        res = minimize(self.average_log_weight, alphas0, method="SLSQP", args=(X, y), constraints=cons, options={'disp': True}) # jac=self.average_log_w_der,
        optimal_alphas = res.x
        return optimal_alphas

    def average_log_weight(self, alphas: np.ndarray, X_p: np.ndarray, Y_p: np.ndarray) -> float:
        """
        Average logged weight over input X_p, Y_p.
        Return single numeric value.
        For optimization purposes: compute max alpha
        """
        assert alphas.shape[0] == self.X_s.shape[0] == self.Y_s.shape[0] , "Alpha required with same dimensions as X_s and Y_s!"
        M = X_p.shape[0]
        _w = self.weight(alphas=alphas, x_p=X_p, y_p=Y_p, X_s=self.X_s, Y_s=self.Y_s, eta=self.eta)
        w = np.sum(np.log(_w)) / M
        return -w # negative s.t. we minimize

    @staticmethod
    def weight(alphas: np.ndarray, x_p: np.ndarray, y_p: np.ndarray, X_s: np.ndarray, Y_s: np.ndarray, eta: float) -> np.ndarray:
        """
        Matrix computation of Eq. (8) from KL-ITL J.Garcke
        """
        assert alphas.shape[0] == X_s.shape[0] == Y_s.shape[0] , "Alpha required with same dimensions as X_s and Y_s!"
        X_s_Y_s = np.hstack([X_s, Y_s])
        x_p_y_p = np.hstack([x_p, y_p])
        x_p_y_p = x_p_y_p if x_p_y_p.shape[0] != x_p_y_p.shape[-1] else x_p_y_p[np.newaxis, :]
        assert X_s_Y_s.shape[-1] == x_p_y_p.shape[-1]
        w = alphas * np.exp(-distance_matrix(x_p_y_p, X_s_Y_s)/(2*(eta**2)))
        assert w.shape[-1] == X_s.shape[0]
        return w

    def get_name(self):
        return f"{self.name}_cv{self.n_splits}_k{self.threshold}" 
        


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
        if missed_indices is not None and len(_X) != len(X):
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


def positional_splitter(seqs, query_seq, val: bool, offset: int, pos_per_fold: int) -> Tuple[list, list, list]:
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
