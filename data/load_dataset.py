import numpy as np
import re 
import warnings
import pandas as pd
import pickle
from typing import Tuple
from os.path import join, dirname
from data.get_alphabet import get_alphabet
from data.load_augmentation import load_augmentation
from util.aa2int import map_alphabets
from util.mlflow.constants import TRANSFORMER, VAE, ONE_HOT, NONSENSE, ESM, EVE, VAE_DENSITY, VAE_AUX, VAE_RAND

base_path = join(dirname(__file__), "files")


def load_one_hot(name: str, desired_alphabet=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load OneHot encoding and observations for input name protein.
    Returns representation and observation vector.
    """
    if name not in ["1FQG", "BRCA", "CALM", "MTH3", "TIMB", "UBQT", "TOXI"]:
        raise ValueError("Unknown dataset: %s" % name)
    if name == "1FQG":  # beta-lactamase
        name = "blat"
    df_name = f"{name.lower()}_data_df"
    X, Y = __load_df(name=df_name, x_column_name="seqs")
    X = X.astype(np.int64)
    if desired_alphabet is not None:
        alphabet_map = map_alphabets(get_alphabet(name), desired_alphabet)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X[i, j] = alphabet_map[X[i, j]]
    return X, Y


def load_transformer(name: str, desired_alphabet=None) -> Tuple[np.ndarray, np.ndarray]:
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
        d = pickle.load(open(join(base_path, "toxi_data_df.pkl"), "rb"))
        idx = np.logical_not(np.isnan(d["assay"]))
        Y = np.vstack(d["assay"].loc[idx])
        X = pickle.load(open(join(base_path, "ProtBert_toxi_labelled_seqs.pkl"), "rb"))
        X = np.vstack(X)
    else:
        raise ValueError("Unknown dataset: %s" % name)
    return X, Y


def load_vae(name: str, vae_suffix: str) -> Tuple[np.ndarray, np.ndarray]:
    if name not in ["1FQG", "BRCA", "CALM", "MTH3", "TIMB", "UBQT", "TOXI"]:
        raise ValueError("Unknown dataset: %s" % name)
    if name == "1FQG":
        name = "blat"
    df_filename = f"{name.lower()}_seq_reps_n_phyla.pkl"
    vae_filename = f"{name.lower()}_{vae_suffix}.pkl"
    d = pickle.load(open(join(base_path, df_filename), "rb"))
    idx = np.logical_not(np.isnan(d["assay"]))
    Y = np.vstack(d["assay"].loc[idx])
    X = pickle.load(open(join(base_path, vae_filename), "rb"))[idx[idx==True].index]
    if name == "toxi": # TODO make X selection more elegant!
        X = np.load(join(base_path, "toxi_VAE_reps.npy"))[idx]
    return X, Y


def load_esm(name: str) -> Tuple[np.ndarray, np.ndarray]:
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
        d = pickle.load(open(join(base_path, "toxi_data_df.pkl"), "rb"))
        idx = np.logical_not(np.isnan(d["assay"]))
        Y = np.vstack(d["assay"].loc[idx])
        X = pickle.load(open(join(base_path, "toxi_esm_rep.pkl"), "rb"))
    else:
        raise ValueError("Unknown dataset: %s" % name)
    return X, Y


def load_eve(name) -> Tuple[np.ndarray, np.ndarray]:
    if name == "1FQG":
        d = pickle.load(open(join(base_path, "blat_seq_reps_n_phyla.pkl"), "rb"))
    elif name == "BRCA":
        d = pickle.load(open(join(base_path, "brca_seq_reps_n_phyla.pkl"), "rb"))
    elif name == "CALM":
        d = pickle.load(open(join(base_path, "calm_seq_reps_n_phyla.pkl"), "rb"))
    elif name == "MTH3":
        d = pickle.load(open(join(base_path, "mth3_seq_reps_n_phyla.pkl"), "rb"))
    elif name == "TIMB":
        d = pickle.load(open(join(base_path, "timb_seq_reps_n_phyla.pkl"), "rb"))
    elif name == "UBQT":
        d = pickle.load(open(join(base_path, "ubqt_seq_reps_n_phyla.pkl"), "rb"))
    elif name == "TOXI":
        d = pickle.load(open(join(base_path, "toxi_data_df.pkl"), "rb"))
    else:
        raise ValueError("Unknown dataset: %s" % name)
    idx = np.logical_not(np.isnan(d["assay"]))
    observed_d = d.loc[idx].copy()
    mut_idx = __get_mutation_idx(name, observed_d)
    observed_d["mutations"] = mut_idx
    Y = np.vstack(d["assay"].loc[idx])
    if name == "1FQG":
        name = "blat"
    if name == "BRCA":
        name = "brca_brct" # this is from deep sequence BRCA
    eve_df = pd.read_csv(join(base_path, f"EVE_{name.upper()}_2000_samples.csv"))
    merged_df_eve_observations = pd.merge(observed_d, eve_df, on="mutations")
    latent_z = merged_df_eve_observations['mean_latent_dim'].tolist()
    if len(latent_z) != len(Y):
        warnings.warn(f"no. EVE mutants {len(latent_z)} != no. observations {len(Y)}! \n Diff: {len(Y)-len(latent_z)} ...")
        Y = np.vstack(merged_df_eve_observations['assay'])
    X = np.array([np.fromstring(seq[1:-1], np.float64, sep=",") for seq in latent_z])
    return X, Y 


def load_nonsense(name) -> Tuple[np.ndarray, np.ndarray]:
    _, Y = load_dataset(name, representation=ONE_HOT)
    restore_seed = np.random.randint(12345)
    np.random.seed(0)
    X = np.random.randn(Y.shape[0], 2)
    np.random.seed(restore_seed)
    return X, Y


def load_dataset(name: str, desired_alphabet=None, representation=ONE_HOT):
    if desired_alphabet is not None and representation is not ONE_HOT:
        raise ValueError("Desired alphabet can only have a value when representation is one hot!")
    if representation is ONE_HOT:
        X, Y = load_one_hot(name, desired_alphabet=desired_alphabet)
    else:
        if representation == TRANSFORMER:
            X, Y = load_transformer(name)
        elif representation == VAE:
            X, Y = load_vae(name, vae_suffix="VAE_reps")
        elif representation == VAE_AUX:
            X, Y = load_vae(name, vae_suffix="AUX_VAE_reps_RANDOM_VAL")
        elif representation == VAE_RAND:
            X, Y = load_vae(name, vae_suffix="VAE_reps_RANDOM_VAL")
        elif representation == ESM:
            X, Y = load_esm(name)
        elif representation == EVE:
            X, Y = load_eve(name)
        elif representation == VAE_DENSITY:
            if name not in ["1FQG", "BRCA", "CALM", "MTH3", "TIMB", "UBQT", "TOXI"]:
                raise ValueError(f"Unknown dataset: {name}")
            X, Y, _ = load_augmentation(name=name, augmentation=VAE_DENSITY)
        elif representation == NONSENSE:
            X, Y = load_nonsense(name)
        else:
            raise ValueError("Unknown value for representation: %s" % representation)
        X = X.astype(np.float64)  # in the one-hot case we want int64, that's why the cast is in this position
    Y = Y.astype(np.float64)
    assert(X.shape[0] == Y.shape[0])
    assert(Y.shape[1] == 1)
    # We flip the sign of Y so our optimization experiment is a minimazation problem
    # litterature review of modelled proteins showed higher values are better
    return X, -Y


def get_wildtype_and_offset(name: str) -> Tuple[str, int]:
    """
    Get encoded wildtype from sequence alignment.
    Get offset with respect to wild-type sequence from alignment FASTA identifier.
    """
    if name == "1FQG":  # beta-lactamase
        d = pickle.load(open(join(base_path, "blat_data_df.pkl"), "rb"))
        sequence_offset = __parse_sequence_offset_from_alignment("alignments/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m")
        wt = d['seqs'][0].astype(np.int64)
    elif name == "BRCA":
        d = pickle.load(open(join(base_path, "brca_data_df.pkl"), "rb"))
        #sequence_offset = __parse_sequence_offset_from_alignment("alignments/BRCA1_HUMAN_1_b0.5.a2m") # This is a different BRCA sequence reference
        sequence_offset = __parse_sequence_offset_from_alignment("alignments/BRCA1_HUMAN_BRCT_1_b0.3.a2m")
        wt = d['seqs'][0].astype(np.int64)
    elif name == "CALM":
        d = pickle.load(open(join(base_path, "calm_data_df.pkl"), "rb"))
        sequence_offset = __parse_sequence_offset_from_alignment("alignments/CALM1_HUMAN_1_b0.5.a2m")
        wt = d['seqs'][0].astype(np.int64)
    elif name == "MTH3":
        d = pickle.load(open(join(base_path, "mth3_data_df.pkl"), "rb"))
        sequence_offset = __parse_sequence_offset_from_alignment("alignments/MTH3_HAEAESTABILIZED_1_b0.5.a2m")
        wt = d['seqs'][0].astype(np.int64)
    elif name == "TIMB":
        d = pickle.load(open(join(base_path, "timb_data_df.pkl"), "rb"))
        sequence_offset = __parse_sequence_offset_from_alignment("alignments/TRPC_THEMA_1_b0.5.a2m")
        wt = d['seqs'][0].astype(np.int64)
    elif name == "UBQT":
        d = pickle.load(open(join(base_path, "ubqt_data_df.pkl"), "rb"))
        sequence_offset = __parse_sequence_offset_from_alignment("alignments/RL401_YEAST_1_b0.5.a2m")
        wt = d['seqs'][0].astype(np.int64)
    elif name == "TOXI":
        d = pickle.load(open(join(base_path, "toxi_data_df.pkl"), "rb"))
        sequence_offset = __parse_sequence_offset_from_alignment("alignments/parEparD_3.a2m")
        wt = np.array(d[d.mutant=="wt"].seqs.values[0]).astype(np.int64)
    else:
        raise ValueError("Unknown dataset: %s" % name)
    return wt, sequence_offset


def __parse_sequence_offset_from_alignment(alignment_filename: str):
    # first line is wildtype, get sequence identifier from fasta alignment
    wt_alignment = open(join(base_path, alignment_filename)).readline().rstrip()
    seq_range = re.findall(r"\b\d+-\d+\b", wt_alignment)[0]
    start_idx = int(seq_range.split("-")[0]) - 1
    return start_idx


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


def __get_mutation_idx(name, df: pd.DataFrame) -> list:
    """
    Derive mutation idx from sequence against available WT.
    Output: list of mutation codes e.g. L3A, A4Y, ...
    """
    wt, seq_offset = get_wildtype_and_offset(name)
    alphabet = get_alphabet(name)
    idx_to_aa_alphabet = dict((v,k) for k,v in alphabet.items())
    wt_seq = "".join([idx_to_aa_alphabet.get(idx) for idx in wt])
    mutation_idx = []
    for mut in df.seqs:
        diff_idx = np.where(wt!=mut)[0]
        m = [] # account for multimutants
        for i in diff_idx:
            seq_str = "".join([idx_to_aa_alphabet.get(idx) for idx in mut])
            from_aa = wt_seq[i]
            to_aa = seq_str[i]
            m_idx = i+1 + seq_offset
            m.append(str(from_aa)+str(m_idx)+str(to_aa))
        mutation_idx.append(":".join(m))
    return mutation_idx


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

