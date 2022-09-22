from algorithms import GMMRegression
from algorithms import GPonRealSpace, KNN, RandomForest
from gpflow.kernels import SquaredExponential, Matern52
from visualization import plot_metric_for_mixtures, plot_metric_against_threshold
from data import PositionSplitter, RandomSplitter
from util.mlflow.convenience_functions import get_mlflow_results_artifacts
from util.mlflow.constants import LINEAR, MEAN_Y, MSE, SPEARMAN_RHO, ONE_HOT, STD_Y, TRANSFORMER, EVE, ESM


def plot_mixtures(datasets: list, algos: list, metrics: list, representations: list, protocol: list, dimensions: int, d_reduction: str):
    results_dict = {}
    for dim in dimensions:
        results_dict[str(dim)] = get_mlflow_results_artifacts(datasets=datasets, reps=representations, metrics=metrics, algos=algos, train_test_splitter=protocol, augmentation=[None],
                                                dim=dim, dim_reduction=d_reduction)
    plot_metric_for_mixtures(results_dict, threshold=0.5, protocol=protocol.get_name())


def plot_performance_against_threshold(datasets: list, algos: list, metrics: list, representations: list, protocol: list, dimension: int, d_reduction: str):
    results_dict = get_mlflow_results_artifacts(datasets=datasets, reps=representations, metrics=metrics, algos=algos, train_test_splitter=protocol, augmentation=[None],
                                            dim=dimension, dim_reduction=d_reduction)
    plot_metric_against_threshold(results_dict, metrics=[MSE, SPEARMAN_RHO], protocol_name=protocol.get_name())


if __name__ == "__main__":
    datasets=["1FQG"] 
    # datasets=["UBQT"]
    protocol=PositionSplitter(datasets[0])
    # protocol=RandomSplitter(datasets[0])
    dimensions=[10, 100, 1000]
    dim_reduction=LINEAR
    algos = [GMMRegression().get_name()]
    reps = [ESM, TRANSFORMER, EVE, ONE_HOT]
    metrics = [MSE, SPEARMAN_RHO, MEAN_Y, STD_Y] # MSE
    # plot_mixtures(datasets=datasets, algos=algos, metrics=metrics, representations=reps, protocol=protocol,
    #             dimensions=dimensions, d_reduction=dim_reduction)
    all_algos = [GPonRealSpace().get_name(), GPonRealSpace(kernel_factory= lambda: SquaredExponential()).get_name(), GPonRealSpace(kernel_factory= lambda: Matern52()).get_name(),
             RandomForest().get_name(), KNN().get_name()]
    plot_performance_against_threshold(datasets=datasets, algos=all_algos, metrics=metrics, representations=reps, protocol=protocol, dimension=None, d_reduction=LINEAR) 