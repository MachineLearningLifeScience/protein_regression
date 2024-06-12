from gpflow.kernels import SquaredExponential

from algorithms import GPonRealSpace, RandomForest
from data.train_test_split import PositionSplitter, RandomSplitter
from util.mlflow.constants import (
    ESM,
    EVE,
    LINEAR,
    MSE,
    ONE_HOT,
    SPEARMAN_RHO,
    TRANSFORMER,
)
from visualization.plot_lowerdim import plot_lower_dim_results

if __name__ == "__main__":
    datasets = ["1FQG", "UBQT", "CALM"]
    algos = [
        GPonRealSpace().get_name(),
        GPonRealSpace(kernel_factory=lambda: SquaredExponential()).get_name(),
        RandomForest().get_name(),
    ]  #
    metrics = [MSE, SPEARMAN_RHO]
    representations = [TRANSFORMER, ESM, ONE_HOT, EVE]
    dimensions = [2, 10, 100, 1000, None]
    dim_reduction = LINEAR  # LINEAR, NON_LINEAR
    cv_types = [RandomSplitter(datasets[0]), PositionSplitter(datasets[0])]
    for dataset in datasets:
        plot_lower_dim_results(
            datasets=[dataset],
            algorithms=algos,
            representations=representations,
            cv_types=cv_types,
            dimensions=dimensions,
            metrics=metrics,
            dim_reduction=dim_reduction,
        )
