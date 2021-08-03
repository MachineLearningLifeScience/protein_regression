import numpy as np
import pickle
from os.path import join, dirname

from util.aa2int import map_alphabets
from util.mlflow.constants import TRANSFORMER, VAE

base_path = join(dirname(__file__), "files")


def load_dataset(name: str, desired_alphabet=None, representation=None):
    if representation is None:
        if name == "1FQG":  # beta-lactamase
            X, Y = __load_df(name="BLAT_data_df", x_column_name="seqs")
        elif name == "BRCA":
            X, Y = __load_df(name="brca_data_df", x_column_name="seqs")
        elif name == "CALM":
            X, Y = __load_df(name="calm_data_df", x_column_name="seqs")
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
        X = X.astype(np.int64)
        if desired_alphabet is not None:
            alphabet_map = map_alphabets(get_alphabet(name), desired_alphabet)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    X[i, j] = alphabet_map[X[i, j]]
    else:
        if desired_alphabet is not None:
            raise ValueError("Representation and desired alphabet MUST NOT have a value at the same time!")
        if representation == TRANSFORMER:
            if name == "1FQG":
                X, Y = __load_df(name="blat_seq_reps_n_phyla", x_column_name="protbert_mean")
            elif name == "BRCA":
                X, Y = __load_df(name="brca_seq_reps_n_phyla", x_column_name="protbert_mean")
            elif name == "CALM":
                X, Y = __load_df(name="calm_seq_reps_n_phyla", x_column_name="protbert_mean")
            else:
                raise ValueError("Unknown dataset: %s" % name)
        elif representation == VAE:
            if name == "1FQG":
                d = pickle.load(open(join(base_path, "blat_seq_reps_n_phyla.pkl"), "rb"))
                idx = np.logical_not(np.isnan(d["assay"]))
                Y = np.vstack(d["assay"].loc[idx])
                X = pickle.load(open(join(base_path, "blat_VAE_reps.pkl"), "rb"))
            elif name == "BRCA":
                d = pickle.load(open(join(base_path, "brca_seq_reps_n_phyla.pkl"), "rb"))
                idx = np.logical_not(np.isnan(d["assay"]))
                Y = np.vstack(d["assay"].loc[idx])
                X = pickle.load(open(join(base_path, "brca_VAE_reps.pkl"), "rb"))
            elif name == "CALM":
                d = pickle.load(open(join(base_path, "calm_seq_reps_n_phyla.pkl"), "rb"))
                idx = np.logical_not(np.isnan(d["assay"]))
                Y = np.vstack(d["assay"].loc[idx])
                X = pickle.load(open(join(base_path, "calm_VAE_reps.pkl"), "rb"))
            else:
                raise ValueError("Unknown dataset: %s" % name)
        else:
            raise ValueError("Unknown value for representation: %s" % representation)
        X = X.astype(np.float64)  # in the one-hot case we want int64, that's why the cast is in this position

    Y = Y.astype(np.float64)
    assert(X.shape[0] == Y.shape[0])
    assert(Y.shape[1] == 1)
    return X, Y


def get_wildtype(name: str):
    if name == "1FQG":  # beta-lactamase
        d = pickle.load(open(join(base_path, "BLAT_data_df.pkl"), "rb"))
        wt = d['seqs'][0].astype(np.int64)
    elif name == "BRCA":
        d = pickle.load(open(join(base_path, "brca_data_df.pkl"), "rb"))
        wt = d['seqs'][0].astype(np.int64)
    elif name == "CALM":
        d = pickle.load(open(join(base_path, "calm_data_df.pkl"), "rb"))
        wt = d['seqs'][0].astype(np.int64)
    else:
        raise ValueError("Unknown dataset: %s" % name)
    return wt


def get_alphabet(name: str):
    if name == "1FQG" or name == "BRCA" or name == "CALM":
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


def __load_df(name: str, x_column_name: str):
    """
    Function to load X, Y columns from Jacob's dataframes.
    :param name:
        name of the dataset
    :param x_column_name:
        name of the column
    :return:
    """
    d = pickle.load(open(join(base_path, "%s.pkl" % name), "rb"))
    idx = np.logical_not(np.isnan(d["assay"]))
    X = np.vstack(d[x_column_name].loc[idx])  #.astype(np.int64)
    Y = np.vstack(d["assay"].loc[idx])
    return X, Y
