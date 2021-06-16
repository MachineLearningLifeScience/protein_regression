import numpy as np
import pickle
from os.path import join, dirname

from util.aa2int import map_alphabets

base_path = join(dirname(__file__), "files")


def load_dataset(name: str, desired_alphabet=None):
    if name == "1FQG":  # beta-lactamase
        d = pickle.load(open(join(base_path, "BLAT_data_df.pkl"), "rb"))
        idx = np.logical_not(np.isnan(d["assay"]))
        X = np.vstack(d["seqs"].loc[idx]).astype(np.int64)
        Y = np.vstack(d["assay"].loc[idx])
    elif name == "DEBUG_SE":
        np.random.seed(42)
        #X = np.random.randn(100, 1)
        X = np.linspace(-3, 3, 100).reshape(-1, 1)
        import tensorflow as tf
        from gpflow.kernels import SquaredExponential
        k = SquaredExponential()
        K = k.K(tf.constant(X)).numpy()
        L = np.linalg.cholesky(K + 1e-6 * np.eye(X.shape[0]))
        Y = L @ np.random.randn(X.shape[0], 1)
    else:
        raise ValueError("Unknown dataset: %s" % name)
    if desired_alphabet is not None:
        alphabet_map = map_alphabets(get_alphabet(name), desired_alphabet)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X[i, j] = alphabet_map[X[i, j]]
    return X, Y


def get_wildtype(name: str):
    if name == "1FQG":  # beta-lactamase
        d = pickle.load(open(join(base_path, "BLAT_data_df.pkl"), "rb"))
        wt = d['seqs'][0].astype(np.int64)
    else:
        raise ValueError("Unknown dataset: %s" % name)
    return wt


def get_alphabet(name: str):
    if name == "1FQG":  # beta-lactamase
        data_alphabet = list(enumerate([
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "<mask>",
            'B'
        ]))
        data_alphabet = {a: i for (i, a) in data_alphabet}
        return data_alphabet
    raise ValueError("Unknown dataset: %s" % name)

