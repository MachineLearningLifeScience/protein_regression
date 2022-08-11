from algorithms import GMMRegression
from visualization import plot_metric_for_mixtures
from data import PositionSplitter, RandomSplitter
from util.mlflow.convenience_functions import get_mlflow_results_artifacts
from util.mlflow.constants import LINEAR, MEAN_Y, MSE, SPEARMAN_RHO, ONE_HOT, STD_Y, TRANSFORMER, EVE, ESM


def plot_mixtures(datasets: list, algos: list, metrics: list, representations: list, protocol: list, dimensions: int, d_reduction: str):
    results_dict = {}
    for dim in dimensions:
        results_dict[str(dim)] = get_mlflow_results_artifacts(datasets=datasets, reps=representations, metrics=metrics, algos=algos, train_test_splitter=protocol, augmentation=[None],
                                                dim=dim, dim_reduction=d_reduction)
    plot_metric_for_mixtures(results_dict, threshold=1.5, protocol=protocol.get_name())


if __name__ == "__main__":
    datasets=["1FQG"] 
    #protocol=PositionSplitter(datasets[0])
    protocol=RandomSplitter(datasets[0])
    dimensions=[2, 10, 100, 1000]
    dim_reduction=LINEAR
    algos = [GMMRegression().get_name()]
    reps = [ESM, TRANSFORMER, EVE, ONE_HOT]
    metrics = [MSE, SPEARMAN_RHO, MEAN_Y, STD_Y] # MSE
    plot_mixtures(datasets=datasets, algos=algos, metrics=metrics, representations=reps, protocol=protocol,
                dimensions=dimensions, d_reduction=dim_reduction)