from algorithms.abstract_algorithm import AbstractAlgorithm
from algorithms.gmm_regression import GMMRegression
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.KNN import KNN
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from algorithms.uncertain_rf import (Uncertain_RandomForestRegressor,
                                     UncertainRandomForest)

NUM_TRAINABLE_PARAMETERS = {"KNN": 1, 
                            "uncertainRF": 100,
                            "RF": 100, # NOTE: this is default configuration, if optimization=True
                            "GPlinear": 2, # theta = {sigma, data_sigma}
                            "GPsqexp": 3, # theta = {sigma, l, data_sigma}
                            "GPmatern52": 3,
                            }