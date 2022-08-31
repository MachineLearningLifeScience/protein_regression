import pytest
from data import load_dataset
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


# def test_loaded_observations_against_original_TOXI():
#     assert False