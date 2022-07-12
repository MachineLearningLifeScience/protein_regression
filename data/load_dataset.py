import numpy as np
import re 
import warnings
import pandas as pd
import pickle
from typing import Tuple, Callable

from os.path import join, dirname
from data.get_alphabet import get_alphabet
from util.aa2int import map_alphabets
from util.mlflow.constants import TRANSFORMER, VAE, ONE_HOT, NONSENSE, ESM, EVE
from util.mlflow.constants import VAE_DENSITY, VAE_AUX, VAE_RAND, EVE_DENSITY, ROSETTA

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


def load_eve(name):
    eve_df = __load_eve_df(name)
    latent_z = eve_df['mean_latent_dim'].tolist()
    Y = np.vstack(eve_df["assay"]) # select assay observations for which merged
    if len(latent_z) != len(Y):
        warnings.warn(f"no. EVE mutants {len(latent_z)} != no. observations {len(Y)}! \n Diff: {len(Y)-len(latent_z)} ...")
        Y = np.vstack(eve_df['assay'])
    X = np.array([np.fromstring(seq[1:-1], np.float64, sep=",") for seq in latent_z])
    return X, Y 


def __load_eve_df(name) -> Tuple[np.ndarray, np.ndarray]:
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
    if name == "1FQG":
        name = "blat"
    if name == "BRCA":
        name = "brca_brct" # this is from deep sequence BRCA
    eve_df = pd.read_csv(join(base_path, f"EVE_{name.upper()}_2000_samples.csv"))
    merged_df_eve_observations = pd.merge(observed_d, eve_df, on="mutations")
    return merged_df_eve_observations


def load_nonsense(name) -> Tuple[np.ndarray, np.ndarray]:
    _, Y = load_dataset(name, representation=ONE_HOT)
    restore_seed = np.random.randint(12345)
    np.random.seed(0)
    X = np.random.randn(Y.shape[0], 2)
    np.random.seed(restore_seed)
    return X, Y


def load_sequences_of_representation(name, representation):
    """
    Assay sequence encodings may have mismatches with loaded representations.
    This function returns the sequences associated with the representation.
    """
    if representation == EVE:
        ref_df = __load_eve_df(name)
    elif representation == EVE_DENSITY:
        if name.upper() == "1FQG":
            name = "BLAT"
        ref_df = __load_eve_mutations_and_observations_df(name)
        ref_df = ref_df.dropna(subset=["assay", "evol_indices"])
    else:
        raise NotImplementedError
    return ref_df.seqs


def get_load_function(representation) -> Callable:
    if representation is ONE_HOT:
        return load_one_hot
    elif representation == TRANSFORMER:
            return load_transformer
    elif "vae" in representation:
        return load_vae
    elif representation == ESM:
        return load_esm
    elif representation == EVE:
        return load_eve
    elif representation == VAE_DENSITY:
        return load_augmentation
    elif representation == EVE_DENSITY:
        return load_augmentation
    elif representation == NONSENSE:
        return load_nonsense


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
        elif representation == VAE_AUX: # THIS IS THE VAE WITH AUXILIARY ELBO NN
            X, Y = load_vae(name, vae_suffix="AUX_VAE_reps_RANDOM_VAL")
        elif representation == VAE_RAND: # THIS IS THE REFERENCE VAE REPRESENTATION
            X, Y = load_vae(name, vae_suffix="VAE_reps_RANDOM_VAL")
        elif representation == ESM:
            X, Y = load_esm(name)
        elif representation == EVE:
            X, Y = load_eve(name)
        elif representation == VAE_DENSITY:
            X, Y, _ = load_augmentation(name=name, augmentation=VAE_DENSITY)
        elif representation == EVE_DENSITY:
            X, Y, _ = load_augmentation(name=name, augmentation=EVE_DENSITY)
        elif representation == NONSENSE:
            X, Y = load_nonsense(name)
        else:
            raise ValueError("Unknown value for representation: %s" % representation)
        X = X.astype(np.float64)  # in the one-hot case we want int64, that's why the cast is in this position
    Y = Y.astype(np.float64)
    assert(X.shape[0] == Y.shape[0])
    assert(Y.shape[1] == 1)
    # We flip the sign of Y so our optimization experiment is a minimazation problem
    # literature review of modelled proteins showed higher values are better
    return X, -Y


def get_wildtype_and_offset(name: str) -> Tuple[str, int]:
    """
    Get encoded wildtype from sequence alignment.
    Get offset with respect to wild-type sequence from alignment FASTA identifier.
    """
    if name == "1FQG" or name == "BLAT":  # beta-lactamase
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
        # special case parD (F7YBW8) anti-toxin: MSA contains F7YBW7 sequence prior, that is 103 long -> offset 103
        sequence_offset += 103
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


def compute_mutations_csv_from_observations(observations_file: str, offset: int=0) -> None:
    """
    Read in observation CSV, parse mutations from observation CSV to 'mutations' column csv, required by EVE.
    DeepSequence reference data had 'mutant' column. ProteinGym has 'mutations' and 'mutant'
    Adjust by optional offset - required to compute TOXI: ParD mutations
    """
    obs_df = pd.read_csv(observations_file, sep=";")
    m_column = 'mutations' if 'mutations' in obs_df.columns else 'mutant'
    mutation_dict = {'mutations': []}
    for mut in obs_df[m_column]:
        if mut == "wt":
            continue
        _mut = mut.split(":") # parse multi-mutants
        mutation = ["".join([m[0], str(int(m[1:-1])+offset), m[-1]]) for m in _mut]
        mutation_dict['mutations'].append(":".join(mutation))
    out_filename = observations_file.split("_")[0] + "_mutations.csv"
    print(f"writing: {out_filename}")
    pd.DataFrame(mutation_dict).to_csv(out_filename, index=False)
    


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


def load_augmentation(name: str, augmentation: str):
    """
    Load data augmentation vectors.
    For Rosetta, VAE densities, EVE evo-scores.
    Data augmentation vectors for VAE and EVE also used as 1D representations.
    Not all assay data has a value due to DeepSequence/EVE setup. Therefore idx_miss is vector of not-computed values.
    Returns:
    A: np.ndarray - augmentation vector
    Y: np.ndarray - assay observations
    idx_miss: np.ndarray - index of missings w.r.t. assay dataframe
    """
    if augmentation == ROSETTA:
        if name not in ["1FQG", "CALM", "UBQT"]:
            raise ValueError("Unknown dataset: %s" % name)
        if name == "1FQG":
            A, Y, idx_miss = _load_rosetta_augmentation(name="BLAT")
        else:
            A, Y, idx_miss = _load_rosetta_augmentation(name=name.upper())
    elif augmentation == VAE_DENSITY:
        if name not in ["1FQG", "CALM", "UBQT"]:
            raise ValueError("Unknown dataset: %s" % name)
        if name == "1FQG":
            name = "BLAT"
        A, Y, idx_miss = _load_vae_augmentation(name=name.upper())
    elif augmentation == EVE_DENSITY:
        if name not in ["1FQG", "CALM", "UBQT", "BRCA", "TIMB", "MTH3", "TOXI"]:
            raise ValueError("Unknown dataset: %s" % name)
        if name == "1FQG":
            name = "BLAT"
        A, Y, idx_miss = _load_eve_augmentation(name=name.upper())
    else:
        raise NotImplementedError(f"Augmentation {augmentation} | {name} not implemented !")
    return A.astype(np.float64) , Y.astype(np.float64), idx_miss


def __load_assay_df(name: str):
    df = pickle.load(open(join(base_path, "{}_data_df.pkl".format(name.lower())), "rb"))
    alphabet = dict((v, k) for k,v in get_alphabet(name=name).items())
    idx_array = np.logical_not(np.isnan(df["assay"]))
    df = df[idx_array]
    wt_sequence, offset = get_wildtype_and_offset(name)
    if "mutant" in df.columns:
        df = df[df.mutant.str.lower() != "wt"]
        if df.mutant.str.contains(":").any():
            df["last_mutation_position"] = df.mutant.apply(lambda x: x.split(":")[-1][1:-1]).astype(int) + offset
            df["mut_aa"] = df.mutant.apply(lambda x: x.split(":")[-1][-1])
        else:
            df["last_mutation_position"] = df.mutant.str.slice(1, -1).astype(int) + offset
            df["mut_aa"] = df.mutant.str.slice(-1)
    else: # translate last mutational variants from sequence information
        df.reset_index(inplace=True)
        # we infer the mutations by reference to its first sequence
        # idx adjusted +1 for comparability 
        df["mutation_idx"] = [[idx+1+offset for idx in np.argwhere(df.seqs[obs] != wt_sequence)[0]] for obs in range(len(df))]
        df["last_mutation_position"] = [int(mut[-1]) if mut else 0 for mut in df.mutation_idx]
        # we use the LAST mutation value (-1) from the index,in case of multi-mutation values in there
        df["mut_aa"] = [alphabet.get(int(seq[idx-1-offset])) for seq, idx in zip(df.seqs, df.last_mutation_position)]
    return df


def _load_rosetta_augmentation(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rosetta_df = __load_rosetta_augmentation_df(name)
    idx_missing = rosetta_df['DDG'].index[rosetta_df['DDG'].apply(np.isnan)]
    rosetta_df = rosetta_df.dropna(subset=["assay", "DDG"])                             
    A = np.vstack(rosetta_df["DDG"])  
    Y = np.vstack(rosetta_df["assay"])
    return A, Y, idx_missing # select only matching data
    

def __load_rosetta_augmentation_df(name: str) -> pd.DataFrame:
    """
    Load persisted DataFrames and join Rosetta simulations on mutations.
    returns 
        X: np.ndarray : DDG simulation values, v-stacked array
        Y: np.ndarray : observation array values
    """
    rosetta_df = pd.read_csv(join(base_path, "{}_single_mutation_rosetta.csv".format(name.lower())))
    rosetta_df["DDG"] = rosetta_df.DDG.astype(float)
    df = __load_assay_df(name)
    joined_df = pd.merge(rosetta_df, df, how="right", left_on=["position", "mut_aa"], 
                                            right_on=["last_mutation_position", "mut_aa"], 
                                            suffixes=('_rosetta', '_assay'))  
    return joined_df

def _load_vae_augmentation(name):
    vae_data = __load_vae_augmentation_df(name)
    assay_df = __load_assay_df(name)
    A = np.vstack(vae_data)
    Y = np.vstack(assay_df["assay"])
    return A, Y, None


def __load_vae_augmentation_df(name: str):
    with open(join(base_path, "vae_results.pkl"), "rb") as infile:
        vae_data = pickle.load(infile).get(name.lower())
    return vae_data


def _load_eve_augmentation(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    eve_augmentation_df = __load_eve_mutations_and_observations_df(name)
    idx_missing = eve_augmentation_df['evol_indices'].index[eve_augmentation_df['evol_indices'].apply(np.isnan)]
    eve_augmentation_df = eve_augmentation_df.dropna(subset=["assay", "evol_indices"])
    # add WT                
    A = np.vstack(eve_augmentation_df['evol_indices'])
    Y = np.vstack(eve_augmentation_df['assay'])
    return A, Y, idx_missing


def __load_eve_mutations_and_observations_df(name: str) -> pd.DataFrame:
    """
    Load EVE dataframe from CSV and index mutation for mutations and observation assay join.
    """
    df = pd.read_csv(join(base_path, f"EVE_{name}_2000_samples.csv"))
    assert df.iloc[0].evol_indices == 0.0
    df = df.iloc[1:] # drop WT - should be zero/NaN for observations and zero for EVE computation
    assay_df = __load_assay_df(name)
    multivariates = df.mutations.str.contains(":").any()
    _, offset = get_wildtype_and_offset(name)
    if multivariates:
        adjusted_mutations = []
        for mutant in assay_df.mutant:
            _m = []
            for mut in mutant.split(":"):
                from_aa, m_idx, to_aa = mut[0], mut[1:-1], mut[-1]
                _m.append(from_aa + str(int(m_idx)+offset)+ to_aa)
            adjusted_mutations.append(":".join(_m))
        assay_df["adjusted_mutations"] = adjusted_mutations
        joined_df = pd.merge(df, assay_df, how="right", left_on=["mutations"], 
                                                right_on=["adjusted_mutations"])
    else:
        df["last_mutation_position"] = df.mutations.str.slice(1, -1).astype(int) + offset
        df["mut_aa"] = df.mutations.str.slice(-1)
        joined_df = pd.merge(df, assay_df, how="right", left_on=["last_mutation_position", "mut_aa"], 
                                                right_on=["last_mutation_position", "mut_aa"])
    return joined_df

