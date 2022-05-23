import numpy as np
import pickle
from os.path import join, dirname
from util.aa2int import map_alphabets
from util.mlflow.constants import TRANSFORMER, VAE, ONE_HOT, NONSENSE, ESM

base_path = join(dirname(__file__), "files")


def load_dataset(name: str, desired_alphabet=None, representation=ONE_HOT):
    if desired_alphabet is not None and representation is not ONE_HOT:
        raise ValueError("Desired alphabet can only have a value when representation is one hot!")

    if representation is ONE_HOT:
        if name == "1FQG":  # beta-lactamase
            X, Y = __load_df(name="blat_data_df", x_column_name="seqs")
        elif name == "BRCA":
            X, Y = __load_df(name="brca_data_df", x_column_name="seqs")
        elif name == "CALM":
            X, Y = __load_df(name="calm_data_df", x_column_name="seqs")
        elif name == "MTH3":
            X, Y = __load_df(name="mth3_data_df", x_column_name="seqs")
        elif name == "TIMB":
            X, Y = __load_df(name="timb_data_df", x_column_name="seqs")
        elif name == "UBQT":
            X, Y = __load_df(name="ubqt_data_df", x_column_name="seqs")
        elif name == "TOXI":
            X, Y = __load_df(name="toxi_data_df", x_column_name="sequence")
        else:
            raise ValueError("Unknown dataset: %s" % name)
        X = X.astype(np.int64)
        if desired_alphabet is not None:
            alphabet_map = map_alphabets(get_alphabet(name), desired_alphabet)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    X[i, j] = alphabet_map[X[i, j]]
    else:
        if representation == TRANSFORMER:
            if name == "1FQG":
                X, Y = __load_df(name="blat_seq_reps_n_phyla", x_column_name="protbert_mean")
            elif name == "BRCA":
                X, Y = __load_df(name="brca_seq_reps_n_phyla", x_column_name="protbert_mean")
            elif name == "CALM":
                X, Y = __load_df(name="calm_seq_reps_n_phyla", x_column_name="protbert_mean")
            elif name == "MTH3":
                X, Y = __load_df(name="mth3_seq_reps_n_phyla", x_column_name="protbert_mean")
            elif name == "TIMB":
                X, Y = __load_df(name="timb_seq_reps_n_phyla", x_column_name="protbert_mean")
            elif name == "UBQT":
                X, Y = __load_df(name="ubqt_seq_reps_n_phyla", x_column_name="protbert_mean")
            elif name == "TOXI":
                raise NotImplementedError("Feature under implementation")
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
            elif name == "MTH3":
                d = pickle.load(open(join(base_path, "mth3_seq_reps_n_phyla.pkl"), "rb"))
                idx = np.logical_not(np.isnan(d["assay"]))
                Y = np.vstack(d["assay"].loc[idx])
                X = pickle.load(open(join(base_path, "mth3_VAE_reps.pkl"), "rb"))
            elif name == "TIMB":
                d = pickle.load(open(join(base_path, "timb_seq_reps_n_phyla.pkl"), "rb"))
                idx = np.logical_not(np.isnan(d["assay"]))
                Y = np.vstack(d["assay"].loc[idx])
                X = pickle.load(open(join(base_path, "timb_VAE_reps.pkl"), "rb"))
            elif name == "UBQT":
                d = pickle.load(open(join(base_path, "ubqt_seq_reps_n_phyla.pkl"), "rb"))
                idx = np.logical_not(np.isnan(d["assay"]))
                Y = np.vstack(d["assay"].loc[idx])
                X = pickle.load(open(join(base_path, "ubqt_VAE_reps.pkl"), "rb"))
            elif name == "TOXI":
                raise NotImplementedError("Feature under implementation")
            else:
                raise ValueError("Unknown dataset: %s" % name)
        elif representation == ESM:
            if name == "1FQG":
                d = pickle.load(open(join(base_path, "blat_seq_reps_n_phyla.pkl"), "rb"))
                idx = np.logical_not(np.isnan(d["assay"]))
                Y = np.vstack(d["assay"].loc[idx])
                X = pickle.load(open(join(base_path, "blat_esm_rep.pkl"), "rb"))
            elif name == "BRCA":
                d = pickle.load(open(join(base_path, "brca_seq_reps_n_phyla.pkl"), "rb"))
                idx = np.logical_not(np.isnan(d["assay"]))
                Y = np.vstack(d["assay"].loc[idx])
                X = pickle.load(open(join(base_path, "brca_esm_rep.pkl"), "rb"))
            elif name == "CALM":
                d = pickle.load(open(join(base_path, "calm_seq_reps_n_phyla.pkl"), "rb"))
                idx = np.logical_not(np.isnan(d["assay"]))
                Y = np.vstack(d["assay"].loc[idx])
                X = pickle.load(open(join(base_path, "calm_esm_rep.pkl"), "rb"))
            elif name == "MTH3":
                d = pickle.load(open(join(base_path, "mth3_seq_reps_n_phyla.pkl"), "rb"))
                idx = np.logical_not(np.isnan(d["assay"]))
                Y = np.vstack(d["assay"].loc[idx])
                X = pickle.load(open(join(base_path, "mth3_esm_rep.pkl"), "rb"))
            elif name == "TIMB":
                d = pickle.load(open(join(base_path, "timb_seq_reps_n_phyla.pkl"), "rb"))
                idx = np.logical_not(np.isnan(d["assay"]))
                Y = np.vstack(d["assay"].loc[idx])
                X = pickle.load(open(join(base_path, "timb_esm_rep.pkl"), "rb"))
            elif name == "UBQT":
                d = pickle.load(open(join(base_path, "ubqt_seq_reps_n_phyla.pkl"), "rb"))
                idx = np.logical_not(np.isnan(d["assay"]))
                Y = np.vstack(d["assay"].loc[idx])
                X = pickle.load(open(join(base_path, "ubqt_esm_rep.pkl"), "rb"))
            elif name == "TOXI":
                d = pickle.load(open(join(base_path, "toxi_data_df.pkl", "rb")))
                idx = np.logical_not(np.isnan(d["assay"]))
                Y = np.vstack(d["assay"].loc[idx])
                X = pickle.load(open(join(base_path, "toxi_esm_rep.pkl", "rb")))
            else:
                raise ValueError("Unknown dataset: %s" % name)
        elif representation == NONSENSE:
            _, Y = load_dataset(name, representation=ONE_HOT)
            restore_seed = np.random.randint(12345)
            np.random.seed(0)
            X = np.random.randn(Y.shape[0], 2)
            np.random.seed(restore_seed)
        else:
            raise ValueError("Unknown value for representation: %s" % representation)
        X = X.astype(np.float64)  # in the one-hot case we want int64, that's why the cast is in this position

    Y = Y.astype(np.float64)
    assert(X.shape[0] == Y.shape[0])
    assert(Y.shape[1] == 1)
    # We flip the sign of Y so our optimization experiment is a minimazation problem
    # litterature review of modelled proteins showed higher values are better
    return X, -Y


def get_wildtype(name: str):
    if name == "1FQG":  # beta-lactamase
        d = pickle.load(open(join(base_path, "blat_data_df.pkl"), "rb"))
        wt = d['seqs'][0].astype(np.int64)
    elif name == "BRCA":
        d = pickle.load(open(join(base_path, "brca_data_df.pkl"), "rb"))
        wt = d['seqs'][0].astype(np.int64)
    elif name == "CALM":
        d = pickle.load(open(join(base_path, "calm_data_df.pkl"), "rb"))
        wt = d['seqs'][0].astype(np.int64)
    elif name == "MTH3":
        d = pickle.load(open(join(base_path, "mth3_data_df.pkl"), "rb"))
        wt = d['seqs'][0].astype(np.int64)
    elif name == "TIMB":
        d = pickle.load(open(join(base_path, "timb_data_df.pkl"), "rb"))
        wt = d['seqs'][0].astype(np.int64)
    elif name == "UBQT":
        d = pickle.load(open(join(base_path, "ubqt_data_df.pkl"), "rb"))
        wt = d['seqs'][0].astype(np.int64)
    else:
        raise ValueError("Unknown dataset: %s" % name)
    return wt


def get_alphabet(name: str):
    if name in ["1FQG", "BLAT", "BRCA", "CALM", "MTH3", "TIMB", "UBQT"]:
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


def make_debug_se_dataset():
    restore_seed = np.random.randint(12345)
    np.random.seed(42)
    # X = np.random.randn(100, 1)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    import tensorflow as tf
    from gpflow.kernels import SquaredExponential
    k = SquaredExponential()
    K = k.K(tf.constant(X)).numpy()
    L = np.linalg.cholesky(K + 1e-6 * np.eye(X.shape[0]))
    Y = L @ np.random.randn(X.shape[0], 1)
    np.random.seed(restore_seed)
    return X, Y
