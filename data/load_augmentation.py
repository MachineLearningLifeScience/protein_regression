import pickle
from os.path import join, dirname
import numpy as np
import pandas as pd
from data import get_alphabet
from util.mlflow.constants import VAE_DENSITY, ROSETTA

base_path = join(dirname(__file__), "files")


def load_augmentation(name: str, augmentation: str):
    # TODO load either VAE, ROSETTA data vector and return
    if augmentation == ROSETTA:
        if name == "1FQG":
            A, Y = __load_rosetta_df(name="BLAT", x_column_name="rosetta_ddg")
        elif name == "CALM":
            A, Y = __load_rosetta_df(name="calm", x_column_name="rosetta_ddg")
        elif name == "UBQT":
            A, Y = __load_rosetta_df(name="ubqt", x_column_name="rosetta_ddg")
        else:
            raise ValueError("Unknown dataset: %s" % name)
    else:
        raise NotImplementedError
    return A.astype(np.float64) , Y.astype(np.float64) 
    
def __load_rosetta_df(name: str, x_column_name: str):
    """
    Load persisted DataFrames and join Rosetta simulations on mutations.
    returns 
        X: np.ndarray : DDG simulation values, v-stacked array
        Y: np.ndarray : observation array values
    """
    rosetta_df = pd.read_csv(join(base_path, "./{}_single_mutation_rosetta.csv".format(name)))
    alphabet = dict((v, k) for k,v in get_alphabet(name=name).items())
    df = pickle.load(open(join(base_path, "%s.pkl" % name), "rb"))
    df["seq_AA"] = [[alphabet.get(int(elem)) for elem in seq] for seq in df.seqs]

    # we infer the mutations by reference to its first sequence
    # idx adjusted +1 for comparability
    wt_sequence = df.seq_AA[0]
    df["mutation_idx"] = [[idx+1 for idx in range(len(wt_sequence)) if df.seq_AA[obs][idx] != wt_sequence[idx]] 
                                  for obs in range(len(df))]

    df["last_mutation_position"] = [0] + [int(mut[-1]) for mut in df.mutation_idx[1:]]

    # we use the LAST mutation value (-1) from the index, as there appear to be multi-mutation values in there
    df["mut_aa"] = [""] + [sequence[(int(mutations[-1])-1)] for sequence, mutations in 
                                zip(df.seq_AA[1:], df.mutation_idx[1:])]
    joined_df = pd.merge(rosetta_df, df, how="right", left_on=["position", "mut_aa"], 
                                            right_on=["last_mutation_position", "mut_aa"]).dropna()

    idx = np.logical_not(np.isnan(joined_df["assay"]))
    A = np.vstack(joined_df[x_column_name].loc[idx])  
    Y = np.vstack(joined_df["assay"].loc[idx])
    return A, Y
