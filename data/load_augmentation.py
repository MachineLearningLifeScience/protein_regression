import pickle
from os.path import join, dirname
import numpy as np
import pandas as pd
from data import get_alphabet
from util.mlflow.constants import VAE_DENSITY, ROSETTA

base_path = join(dirname(__file__), "files")


def load_augmentation(name: str, augmentation: str):
    if augmentation == ROSETTA:
        if name == "1FQG":
            A, Y, idx_miss = __load_rosetta_df(name="BLAT")
        elif name == "CALM":
            A, Y, idx_miss = __load_rosetta_df(name="CALM")
        elif name == "UBQT":
            A, Y, idx_miss = __load_rosetta_df(name="UBQT")
        else:
            raise ValueError("Unknown dataset: %s" % name)
    elif augmentation == VAE_DENSITY:
        if name == "1FQG":
            A, Y, idx_miss = __load_vae_df(name="BLAT")
        elif name == "CALM":
            A, Y, idx_miss = __load_vae_df(name="CALM")
        elif name == "UBQT":
            A, Y, idx_miss = __load_vae_df(name="UBQT")
        else:
            raise ValueError("Unknown dataset: %s" % name)
    else:
        raise NotImplementedError
    return A.astype(np.float64) , -Y.astype(np.float64), idx_miss


def __load_assay_df(name: str):
    alphabet = dict((v, k) for k,v in get_alphabet(name=name).items())
    df = pickle.load(open(join(base_path, "{}_data_df.pkl".format(name.lower())), "rb"))
    idx_array = np.logical_not(np.isnan(df["assay"]))
    idx_array[0] = True # include WT for mutation computation
    df = df[idx_array]
    df["seq_AA"] = [[alphabet.get(int(elem)) for elem in seq] for seq in df.seqs]
    # we infer the mutations by reference to its first sequence
    # idx adjusted +1 for comparability 
    wt_sequence = df.seq_AA[0]
    df = df.iloc[1:, :].reset_index() # drop WT again
    df["mutation_idx"] = [[idx+1 for idx in range(len(wt_sequence)) if df.seq_AA[obs][idx] != wt_sequence[idx]] 
                                  for obs in range(len(df))]
    df["last_mutation_position"] = [int(mut[-1]) if mut else 0 for mut in df.mutation_idx]
    # we use the LAST mutation value (-1) from the index,in case of multi-mutation values in there
    df["mut_aa"] = [sequence[(int(mutations[-1])-1)] for sequence, mutations in 
                                zip(df.seq_AA, df.mutation_idx)]
    return df


def __load_rosetta_df(name: str):
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
    idx_missing = joined_df['DDG'].index[joined_df['DDG'].apply(np.isnan)]
    joined_df = joined_df.dropna(subset=["assay", "DDG"])                             
    A = np.vstack(joined_df["DDG"])  
    Y = np.vstack(joined_df["assay"])
    return A, -Y, idx_missing # select only matching data


def __load_vae_df(name: str):
    with open(join(base_path, "vae_results.pkl"), "rb") as infile:
        vae_data = pickle.load(infile).get(name.lower())
    assay_df = __load_assay_df(name)
    A = np.vstack(vae_data)
    Y = np.vstack(assay_df["assay"])
    return A, -Y, None
