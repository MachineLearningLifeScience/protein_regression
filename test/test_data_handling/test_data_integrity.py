import pytest
from data import load_dataset
from data.load_dataset import load_one_hot
from data.load_dataset import __load_df
import numpy as np
from util.mlflow.constants import ONE_HOT, EVE, ESM, TRANSFORMER


@pytest.mark.parametrize("dataset", ["1FQG", "UBQT", "CALM", "TIMB", "BRCA", "MTH3", "TOXI"])
def test_loading_representation_observations_equal_BLAT(dataset):
    _, y_oh = load_dataset(name=dataset, representation=ONE_HOT)
    # _, y_eve = load_dataset(name=dataset, representation=EVE) # NOTE: some eve observations are missing!
    _, y_esm = load_dataset(name=dataset, representation=ESM)
    _, y_tr = load_dataset(name=dataset, representation=TRANSFORMER)
    np.testing.assert_equal(y_esm, y_tr)
    np.testing.assert_equal(y_oh, y_esm)


@pytest.mark.parametrize("dataset", ["1FQG", "UBQT", "CALM", "TIMB", "BRCA", "MTH3", "TOXI"])
def test_loaded_observations_against_data_DF(dataset):
    df_name = dataset if dataset != "1FQG" else "blat" 
    df_name = f"{df_name.lower()}_data_df"
    _, y_df = __load_df(df_name, "seqs")
    _, y_load = load_dataset(dataset, representation=ONE_HOT)
    np.testing.assert_equal(-y_df, y_load)


def test_toxi_sequences_equal_OH_rep():
    toxi_sequences, toxi_obs_seq = load_one_hot("TOXI")
    toxi_data, toxi_obs = load_dataset("TOXI")
    assert toxi_sequences.shape[0] == toxi_data.shape[0]
    np.testing.assert_equal(toxi_obs_seq, -toxi_obs) # NOTE: loading data inverts observations.