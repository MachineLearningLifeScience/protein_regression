import os
import pickle
import pandas as pd
import numpy as np
from itertools import product
from data.train_test_split import AbstractTrainTestSplitter, FractionalRandomSplitter
from gpflow.kernels import SquaredExponential
from algorithms import GPonRealSpace, RandomForest, KNN
from data.train_test_split import RandomSplitter, BlockPostionSplitter, PositionSplitter, FractionalRandomSplitter, BioSplitter
from util.mlflow.convenience_functions import get_mlflow_results, get_mlflow_results_artifacts
from util.mlflow.constants import EVE_DENSITY, ONE_HOT, TRANSFORMER, VAE, ESM, VAE_AUX, VAE_RAND
from util.mlflow.constants import GP_LEN, GP_L_VAR, GP_VAR, MLL, MSE, SPEARMAN_RHO


def extract_parameters(datasets: list, algos: list, reps: list, metrics: list, train_test_splitter: AbstractTrainTestSplitter, ):
    """
    NOTE: extracting and visualizing parameters (tables and figures) is part of the MlFlow results UI. This method is therefore deprecated.
    """
    results_dict = get_mlflow_results(datasets, algos, reps, metrics, train_test_splitter)
    cv_name = train_test_splitter(datasets[0]).get_name()
    sub_metrics = ["mu", "std"]
    header_multi_index = pd.MultiIndex.from_tuples(((a, m, sm) for a in results_dict.get(datasets[0]).keys() 
                                                        for m in metrics for sm in sub_metrics), 
                                            names=["algo", "metric", "sub_m"])
    multi_index = pd.MultiIndex.from_tuples(((d,r, cv_name) for d in datasets for r in reps), 
                                            names=["data", "representation", "cv"])
    df = pd.DataFrame(columns=header_multi_index, index=multi_index)
    for alg in results_dict.get(datasets[0]).keys():
        for metric in metrics:
            for dat in datasets:
                for rep in reps:
                    entry = results_dict.get(dat).get(alg).get(rep).get(None).get(metric)
                    df.loc[(dat, rep, cv_name), (alg, metric)] = (np.round(np.mean(entry), 2), np.round(np.std(entry), 2))
    return df


def extract_cv_information(datasets: list, representations: list, protocols: list, cached_results=True) -> pd.DataFrame:
    algos = [GPonRealSpace().get_name()] # splits are the same across all algos
    cached_filename = f"/Users/rcml/protein_regression/results/cache/metric_comparison_d={'_'.join(datasets)}_a={'_'.join(algos)}_r={'_'.join(representations)}_s={'_'.join([s.get_name() for s in protocols[:5]])}.pkl"
    if cached_results and os.path.exists(cached_filename):
        with open(cached_filename, "rb") as infile:
            mlflow_results = pickle.load(infile)
    else:
        mlflow_results = {}
        for protocol in protocols:
            mlflow_protocol_results = get_mlflow_results_artifacts(datasets=datasets, algos=algos, reps=representations, metrics=[None], train_test_splitter=protocol) # TODO: what metrics?
            mlflow_results[protocol.get_name()] = mlflow_protocol_results
        if cached_results:
            with open(cached_filename, "wb") as outfile:
                pickle.dump(mlflow_results, outfile)
    protocol_names = [protocol.get_name() for protocol in protocols]
    results_iterator = product(datasets, representations, protocol_names)
    header_multi_index = pd.MultiIndex.from_tuples((d, r) for d, r in product(datasets, representations))
    row_multi_index = pd.MultiIndex.from_tuples((p,idx) for p, idx in product(protocol_names, [0]))
    df = pd.DataFrame(columns=header_multi_index, index=row_multi_index)
    for d, r, p in results_iterator:
        entries = mlflow_results.get(p).get(d).get(algos[0]).get(r).get(None)
        if entries:
            for e in entries:
                n_train = len(entries[e].get("train_trues"))
                n_test = len(entries[e].get("trues"))
                df.loc[(p,e), (d,r)] = f"[{n_train};{n_test}]"
        else:
            df.loc[(p, 0), (d,r)] = np.nan
    return df


if __name__ == "__main__":
    ### MAKE CV TABLE
    datasets=["1FQG",  "UBQT", "TIMB", "MTH3", "BRCA"]
    representations = [ONE_HOT, EVE_DENSITY] # splits are the same for ESM/ProtBert/OH , different for EVE derivative
    train_test_splitter=[RandomSplitter("1FQG"), PositionSplitter("1FQG")] # NOTE: fractional splitter is not included here
    cv_df = extract_cv_information(datasets=datasets, representations=representations, protocols=train_test_splitter)
    # main overview
    print(cv_df.to_latex(bold_rows=True, multicolumn_format="r"))
    # biosplitter overview - separate table
    cv_df = extract_cv_information(datasets=["TOXI"], representations=[ONE_HOT], 
                protocols=[BioSplitter("TOXI", 1, 1), BioSplitter("TOXI", 1, 2), BioSplitter("TOXI", 2, 2), BioSplitter("TOXI", 2, 3), BioSplitter("TOXI", 3, 3), BioSplitter("TOXI", 3, 4)])
    print(cv_df.to_latex(bold_rows=True, multicolumn_format="r"))
