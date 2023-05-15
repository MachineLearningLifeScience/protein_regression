import numpy as np
import re 
import warnings
import pandas as pd
import pickle
from typing import Tuple, Callable
from os.path import join, dirname
from data.get_alphabet import get_alphabet
from util import numpy_one_hot_2dmat
from util.aa2int import map_alphabets
from util.mlflow.constants import TRANSFORMER, VAE, ONE_HOT, NONSENSE, ESM, EVE
from util.mlflow.constants import VAE_DENSITY, VAE_AUX, VAE_RAND, EVE_DENSITY, ROSETTA

base_path = join(dirname(__file__), "files")


def get_mutation_diff(seq_x: np.ndarray, seq_y: np.ndarray) -> int:
    """
    compute the number of positions different between two label-encoded arrays.
    """
    diff = np.sum((seq_x - seq_y) != 0)
    return diff


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


# def load_vae(name: str, vae_suffix: str) -> Tuple[np.ndarray, np.ndarray]:
#     if name not in ["1FQG", "BRCA", "CALM", "MTH3", "TIMB", "UBQT", "TOXI"]:
#         raise ValueError("Unknown dataset: %s" % name)
#     if name == "1FQG":
#         name = "blat"
#     df_filename = f"{name.lower()}_seq_reps_n_phyla.pkl"
#     vae_filename = f"{name.lower()}_{vae_suffix}.pkl"
#     d = pickle.load(open(join(base_path, df_filename), "rb"))
#     idx = np.logical_not(np.isnan(d["assay"]))
#     Y = np.vstack(d["assay"].loc[idx])
#     X = pickle.load(open(join(base_path, vae_filename), "rb"))[idx[idx==True].index]
#     if name == "toxi": # TODO make X selection more elegant!
#         X = np.load(join(base_path, "toxi_VAE_reps.npy"))[idx]
#     A = None # TODO: compute A
#     return X, Y, A


def __compute_observation_and_deduplication_indices(df) -> Tuple[np.ndarray, np.ndarray]:
    """
    Required to filter persisted dataframes by indices.
    Filter *_phyla.pkl dataframe by observation index AND deduplication index,
    Filter *_esm_rep.pkl by deduplication
    returns:
    observation_idx: np.ndarray ,
    representation_idx: np.ndarray 
    """
    idx = np.logical_not(np.isnan(df["assay"])) # observations by assay values
    observations_start: int = np.argwhere(idx.values==True)[0][0] # first observation index
    if "mutant" in df.columns:
        duplicated_index = df[["mutant", "assay"]].duplicated()
    else:
        duplicated_index = df[["seqs_aa", "assay"]].duplicated() # find duplicates as sequence+assay repeats
    observation_idx = np.logical_and(idx, ~duplicated_index) # combine observation AND INVERT duplicate for later selection
    anchored_duplicated_index = duplicated_index.set_axis(duplicated_index.index-observations_start) # adjust index by first observation
    representation_index = np.invert(anchored_duplicated_index[anchored_duplicated_index.index >= 0])
    return observation_idx.values, representation_index.values


def load_esm(name: str) -> Tuple[np.ndarray, np.ndarray]:
    if name == "1FQG":
        d = pickle.load(open(join(base_path, "blat_seq_reps_n_phyla.pkl"), "rb"))
        observation_idx, representation_idx = __compute_observation_and_deduplication_indices(d)
        d = d.loc[observation_idx]
        Y = np.vstack(d["assay"])
        X = pickle.load(open(join(base_path, "blat_esm_rep.pkl"), "rb"))[representation_idx]
    elif name == "BRCA":
        d = pickle.load(open(join(base_path, "brca_seq_reps_n_phyla.pkl"), "rb"))
        observation_idx, representation_idx = __compute_observation_and_deduplication_indices(d)
        d = d.loc[observation_idx]
        Y = np.vstack(d["assay"])
        X = pickle.load(open(join(base_path, "brca_esm_rep.pkl"), "rb"))[representation_idx]
    elif name == "CALM":
        d = pickle.load(open(join(base_path, "calm_seq_reps_n_phyla.pkl"), "rb"))
        observation_idx, representation_idx = __compute_observation_and_deduplication_indices(d)
        d = d.loc[observation_idx]
        Y = np.vstack(d["assay"])
        X = pickle.load(open(join(base_path, "calm_esm_rep.pkl"), "rb"))[representation_idx]
    elif name == "MTH3":
        d = pickle.load(open(join(base_path, "mth3_seq_reps_n_phyla.pkl"), "rb"))
        observation_idx, representation_idx = __compute_observation_and_deduplication_indices(d)
        d = d.loc[observation_idx]
        Y = np.vstack(d["assay"])
        X = pickle.load(open(join(base_path, "mth3_esm_rep.pkl"), "rb"))[representation_idx]
    elif name == "TIMB":
        d = pickle.load(open(join(base_path, "timb_seq_reps_n_phyla.pkl"), "rb"))
        observation_idx, representation_idx = __compute_observation_and_deduplication_indices(d)
        d = d.loc[observation_idx]
        Y = np.vstack(d["assay"])
        X = pickle.load(open(join(base_path, "timb_esm_rep.pkl"), "rb"))[representation_idx]
    elif name == "UBQT":
        d = pickle.load(open(join(base_path, "ubqt_seq_reps_n_phyla.pkl"), "rb"))
        observation_idx, representation_idx = __compute_observation_and_deduplication_indices(d)
        d = d.loc[observation_idx]
        Y = np.vstack(d["assay"])
        X = pickle.load(open(join(base_path, "ubqt_esm_rep.pkl"), "rb"))[representation_idx]
    elif name == "TOXI":
        d = pickle.load(open(join(base_path, "toxi_data_df.pkl"), "rb"))
        observation_idx, representation_idx = __compute_observation_and_deduplication_indices(d)
        d = d.loc[observation_idx]
        Y = np.vstack(d["assay"])
        X = pickle.load(open(join(base_path, "toxi_esm_rep.pkl"), "rb"))[representation_idx[observation_idx==True]]
    else:
        raise ValueError("Unknown dataset: %s" % name)
    assert X.shape[0] == Y.shape[0]
    return X, Y


def load_eve(name):
    eve_df = __load_eve_df(name)
    latent_z = eve_df['mean_encoder'].tolist()
    Y = np.vstack(eve_df["assay"]) # select assay observations for which merged
    if len(latent_z) != len(Y):
        warnings.warn(f"no. EVE mutants {len(latent_z)} != no. observations {len(Y)}! \n Diff: {len(Y)-len(latent_z)} ...")
        Y = np.vstack(eve_df['assay'])
    X = np.array([np.fromstring(seq[1:-1], np.float64, sep=",") for seq in latent_z])
    A = eve_df['evol_indices'].to_numpy()
    S = np.vstack(eve_df.seqs).astype(int)
    return S, X, Y, A


def __load_eve_df(name) -> Tuple[np.ndarray, np.ndarray]:
    name = name.upper()
    if name == "1FQG" or name == "BLAT":
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
    observed_d = observed_d.drop_duplicates(subset=["mutations", "assay"])
    _, offset = get_wildtype_and_offset(name)
    if name == "1FQG":
        name = "blat"
    if name == "BRCA":
        name = "brca_brct" # this is from deep sequence BRCA
    eve_df = pd.read_csv(join(base_path, f"EVE_{name.upper()}_2000_samples.csv")).drop_duplicates()
    merged_df_eve_observations = pd.merge(observed_d, eve_df, on="mutations", validate="one_to_one")
    wt = merged_df_eve_observations[merged_df_eve_observations.mutations=="wt"]
    merged_df_eve_observations = merged_df_eve_observations.drop(index=wt.index) if wt is not None else merged_df_eve_observations
    # add mutation positions and AA to DF
    if merged_df_eve_observations.mutations.str.contains(":").any():
        merged_df_eve_observations["last_mutation_position"] = merged_df_eve_observations.mutations.apply(lambda x: x.split(":")[-1][1:-1]).astype(int) + offset
        merged_df_eve_observations["mut_aa"] = merged_df_eve_observations.mutations.apply(lambda x: x.split(":")[-1][-1])
    else:
        merged_df_eve_observations["last_mutation_position"] = merged_df_eve_observations.mutations.str[1:-1].astype(int) + offset
        merged_df_eve_observations["mut_aa"] = merged_df_eve_observations.mutations.str[-1]
    merged_df_eve_observations = pd.concat([merged_df_eve_observations, wt])
    return merged_df_eve_observations


def load_nonsense(name) -> Tuple[np.ndarray, np.ndarray]:
    _, Y = load_dataset(name, representation=ONE_HOT)
    restore_seed = np.random.randint(12345)
    np.random.seed(0)
    X = np.random.randn(Y.shape[0], 2)
    np.random.seed(restore_seed)
    return X, Y


def load_sequences_of_representation(name, representation, augmentation=None):
    """
    Assay sequence encodings may have mismatches with loaded representations. 
    Case: UBQT+eve density, where fewer observations than eve values exist
    This function returns the sequences associated with the representation.
    """
    if representation == EVE or augmentation == EVE:
        ref_df = __load_eve_df(name)
        X = np.vstack(ref_df.seqs)
    elif representation == EVE_DENSITY or augmentation == EVE_DENSITY:
        if name.upper() == "1FQG":
            name = "BLAT"
        ref_df = __load_eve_df(name)
        ref_df = ref_df.dropna(subset=["assay", "evol_indices"])
        X = np.vstack(ref_df.seqs)
    else:
        X, _ = load_one_hot(name)
    return X


def load_dataset(name: str, desired_alphabet=None, representation=ONE_HOT, augmentation=None) -> tuple:
    if desired_alphabet is not None and representation is not ONE_HOT:
        raise ValueError("Desired alphabet can only have a value when representation is one hot!")
    # Representation Loading
    if representation == ONE_HOT:
        X, Y = load_one_hot(name, desired_alphabet=desired_alphabet)
        X = numpy_one_hot_2dmat(X, max=len(get_alphabet(name)))
        # normalize by sequence length
        X = X / X.shape[1]
    else:
        if representation == TRANSFORMER:
            X, Y = load_transformer(name)
        # elif representation == VAE:
        #     X, Y, A = load_vae(name, vae_suffix="VAE_reps")
        # elif representation == VAE_AUX: # VAE WITH AUXILIARY ELBO NN
        #     X, Y, A = load_vae(name, vae_suffix="AUX_VAE_reps_RANDOM_VAL")
        elif representation == ESM:
            X, Y = load_esm(name)
        elif representation == EVE:
            eve_S, X, Y, A = load_eve(name)
            missed_assay_indices = None # EVE protocol covers all possible single variants
        elif representation == EVE_DENSITY: # TODO: add VAE_Density representation again
            eve_S, _, Y, X = load_eve(name)
            X = X[:, np.newaxis]
            missed_assay_indices = None
        elif representation == NONSENSE:
            X, Y = load_nonsense(name)
        else:
            raise ValueError("Unknown value for representation: %s" % representation)
        X = X.astype(np.float64)  # in the one-hot case we want int64, that's why the cast is in this position
    Y = Y.astype(np.float64)
    # Augmentation Loading: 
    if augmentation == ROSETTA or (augmentation == EVE_DENSITY and representation is not EVE):
        S, A, = load_augmentation(name, augmentation)
        data_S = load_one_hot(name, desired_alphabet=desired_alphabet)[0] if representation != EVE else eve_S # EVE sequences are distinct from base data sequences
        X, Y, A, missed_assay_indices = join_X_A_by_available_sequences(X, Y, data_S=data_S, S=S, A=A, name=name)
    assert(Y.shape[1] == 1)
    assert X.shape[0] == Y.shape[0]
    # We flip the sign of Y so our optimization experiment is a minimization problem
    # lit.-review of proteins: higher values are better
    if augmentation:
        return X, (-Y, missed_assay_indices)
    return X, -Y


def load_augmentation(name:str, augmentation: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load requested augmentation. Available either Rosetta or EVE vectors.
    TODO: extend to other VAE vectors.
    Return Tuple: S: sequence for which vector was computed, A: augmentation vector
    """
    if augmentation == ROSETTA:
        S, A = _load_rosetta_augmentation(name)
    elif augmentation == EVE_DENSITY:
        S, _, _, A = load_eve(name)
    else:
        raise NotImplementedError(f"Requested augmentation {augmentation} does not exist!")
    return S, A


def join_X_A_by_available_sequences(X: np.ndarray, Y: np.ndarray, data_S: np.ndarray, S: np.ndarray, A: np.ndarray, name: str) -> tuple:
    """
    Filters representation and entries by available sequences,
    given augmentation sequences S,
    Input: 
        X: merged data-matrix,
        Y: observations,
        data_S: sequences from data-loading
        S: augmentation sequences
        A: augmentation vector
        name: str name
    Returns:
        X: merged and filtered data matrix
        Y: filtered observations
        A: augmentation vector
        missing_assay_indices: mismatch indices between loaded data and augmentation for later splitting protocol
    """
    missed_assay_indices = None
    A = A if A.shape[-1] == 1 else A[:, np.newaxis] # add augmentation vector from Rosetta | EVE | VAE to data-matrix
    # intersection of observed sequences and augmentation vector
    _lh_sequences, _rh_sequences = (data_S, S) if len(data_S) >= len(S) else (S, data_S)
    idx_join = np.in1d([str(_os) for _os in _lh_sequences], [str(_s) for _s in _rh_sequences]) # str representation required for "in" assessment
    if X.shape[0] > A.shape[0]: 
        X = X[idx_join]
        Y = Y[idx_join]
    else:
        A = A[idx_join]
    A /= X.shape[1] # normalize by length of sequence
    X = np.concatenate([X, A], axis=1)
    if np.isnan(X).any():
        warnings.warn(f"NaN values encountered in {name} augmentation! Shape: {X.shape}")
        missed_assay_indices = np.where(np.isnan(X).sum(axis=1).astype(bool))[0]
        not_nan_idx = np.setdiff1d(np.arange(X.shape[0]), missed_assay_indices)
        X = X[not_nan_idx]
        Y = Y[not_nan_idx]
        A = A[not_nan_idx]
        warnings.warn(f"Removing entries... => Shape: {X.shape}")
    assert len(A) == len(Y) == len(X)
    return X, Y, A, missed_assay_indices


def get_wildtype_and_offset(name: str) -> Tuple[str, int]:
    """
    Get encoded wildtype from sequence alignment.
    Get offset with respect to wild-type sequence from alignment FASTA identifier.
    """
    name = name.upper()
    if name == "1FQG" or name == "BLAT":  # beta-lactamase
        d = pickle.load(open(join(base_path, "blat_data_df.pkl"), "rb"))
        sequence_offset = __parse_sequence_offset_from_alignment("alignments/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m")
        wt = d['seqs'][0].astype(np.int64)
    elif name == "BRCA" or name == "BRCA_BRCT":
        d = pickle.load(open(join(base_path, "brca_data_df.pkl"), "rb"))
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
    """
    Utility function to get reference sequence index from FASTA alignment file.
    """
    # first line is wildtype, get sequence identifier from fasta alignment
    wt_alignment = open(join(base_path, alignment_filename)).readline().rstrip()
    seq_range = re.findall(r"\b\d+-\d+\b", wt_alignment)[0]
    start_idx = int(seq_range.split("-")[0]) - 1
    return start_idx


def compute_mutations_csv_from_observations(observations_file: str, offset: int=0) -> None:
    """
    Utility Function as required by EVE.
    Read in observation CSV, parse mutations from observation CSV to 'mutations' column csv.
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
    


def __load_df(name: str, x_column_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to load X, Y columns from pickled/persisted dataframes.
    :param name:
        name of the dataset
    :param x_column_name:
        name of the column
    :return:
    """
    d = pickle.load(open(join(base_path, "%s.pkl" % name), "rb"))
    idx = np.logical_not(np.isnan(d["assay"]))
    d = d.loc[idx]
    d["mutations"] = __get_mutation_idx(name=name.split("_")[0].upper(), df=d)
    d = d.drop_duplicates(subset=["mutations", "assay"])
    X = np.vstack(d[x_column_name])
    Y = np.vstack(d["assay"]) 
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
        mutation = ":".join(m)
        if all(wt==mut) and not any(m):
            mutation = "wt"
        mutation_idx.append(mutation)
    # TODO: resolve below
    # assert len(np.unique(mutation_idx)) == df.seqs.shape[0] "Mismatch between unique mutations and number of input sequences"
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


def __load_assay_df(name: str):
    name = name.lower() if not name.lower() == "1fqg" else "BLAT"
    df = pickle.load(open(join(base_path, f"{name}_data_df.pkl"), "rb"))
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


def _load_rosetta_augmentation(name: str) -> Tuple[np.ndarray, np.ndarray]:
    name = name.lower() if not name.lower() == "1fqg" else "blat"
    rosetta_df = __load_rosetta_augmentation_df(name) 
    S = np.vstack(rosetta_df["seqs"]).astype(int)                        
    A = np.vstack(rosetta_df["DDG"])  
    return S, A


def __load_rosetta_augmentation_df(name: str) -> pd.DataFrame:
    """
    Load persisted DataFrames and join Rosetta simulations on mutations.
    returns 
        X: np.ndarray : DDG simulation values, v-stacked array
        Y: np.ndarray : observation array values
    """
    rosetta_df = pd.read_csv(join(base_path, "{}_single_mutation_rosetta.csv".format(name.lower())))
    wt_sequence, offset = get_wildtype_and_offset(name)
    if offset != 0. and rosetta_df.position.astype(int)[0] == 1:
        warnings.warn(f"{name} Rosetta positions require offset += {offset}!")
    rosetta_df['position'] = rosetta_df.position.astype(int)+offset
    rosetta_df["DDG"] = rosetta_df.DDG.astype(float)
    # if rep == EVE: # special case: does not guarantee same data loaded for representations, due to EVE-internal df construction
    #     observations_df = __filter_eve_by_observations_join(name)
    # else:
    observations_df = __load_assay_df(name)
    observations_df = observations_df[["seqs", "assay", "last_mutation_position", "mut_aa"]].drop_duplicates(subset=["assay", "last_mutation_position", "mut_aa"])
    joined_df = pd.merge(rosetta_df, observations_df, how="right", left_on=["position", "mut_aa"], 
                        right_on=["last_mutation_position", "mut_aa"], 
                        suffixes=('_rosetta', '_assay'), validate="one_to_one")  
    return joined_df


# def __load_eve_mutations_and_observations_df(name: str) -> pd.DataFrame:
#     """
#     Load EVE dataframe from CSV and index mutation for mutations and observation assay join.
#     """
#     assay_df = __load_assay_df(name)
#     assay_df = assay_df.drop_duplicates(subset=["last_mutation_position", "mut_aa", "assay"])
#     if name.upper() == "BRCA":
#         name = "BRCA_BRCT"
#     if name.upper() == "1FQG":
#         name = "BLAT"
#     df = pd.read_csv(join(base_path, f"EVE_{name}_2000_samples.csv"))
#     assert df.iloc[0].evol_indices == 0.0 # Test WT is present
#     df = df.iloc[1:] # drop WT - should be zero/NaN for observations and zero for EVE computation
#     multivariates = df.mutations.str.contains(":").any()
#     _, _offset = get_wildtype_and_offset(name)
#     if multivariates: # TODO: test multivariate case for TOXI with current loading
#         adjusted_mutations = []
#         for mutant in assay_df.mutant:
#             _m = []
#             for mut in mutant.split(":"):
#                 from_aa, m_idx, to_aa = mut[0], mut[1:-1], mut[-1]
#                 _m.append(from_aa + str(int(m_idx)+_offset)+ to_aa) # offset required for TOXI
#             adjusted_mutations.append(":".join(_m))
#         assay_df["adjusted_mutations"] = adjusted_mutations
#         joined_df = pd.merge(df, assay_df, how="right", left_on=["mutations"], 
#                                                 right_on=["adjusted_mutations"], validate="one_to_one")
#     else:
#         df["last_mutation_position"] = df.mutations.str.slice(1, -1).astype(int)
#         df["mut_aa"] = df.mutations.str.slice(-1)
#         joined_df = pd.merge(df, assay_df, how="right", left_on=["last_mutation_position", "mut_aa"], 
#                                                 right_on=["last_mutation_position", "mut_aa"], validate="one_to_one")
#     return joined_df

