import numpy as np
from scipy import stats
from sklearn.metrics import zero_one_loss, mean_squared_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from algorithms import GMMRegression
from util.mlflow.constants import MSE, SPEARMAN_RHO


def plot_metric_for_mixtures(
    results: dict,
    threshold: float,
    protocol: str,
    method=GMMRegression(),
    augmentation=None,
    split=0,
) -> None:
    """
    Plot mixture models results as multiplot (representations[y, X] x dim_PCA)
    Mark Gaussians by color with mean and standard deviation for each.
    Compute Loss respective measurement threshold: cluster = bool(y > threshold)
    """
    # color by cluster
    # plot two rows: observations clustered, representations clustered, columns are dimensions (d=2,10,100,1000)
    dimensions = list(results.keys())
    dataset = list(results.get(dimensions[0]).keys())
    representations = list(
        results.get(dimensions[0]).get(dataset[0]).get(method.get_name()).keys()
    )
    fig, ax = plt.subplots(
        3 * len(representations),
        len(dimensions),
        figsize=(25, 25),
        gridspec_kw={"height_ratios": [2, 1, 0.5] * len(representations)},
    )  # for each representation two rows
    mean_colors = ["purple", "darkorange"]
    marker_colors = ["darkblue", "darkred"]
    for i, d in enumerate(dimensions):
        for j, rep in enumerate(representations):
            j *= 3  # in steps of two
            observations = (
                results.get(d)
                .get(dataset[0])
                .get(method.get_name())
                .get(rep)
                .get(augmentation)
                .get(split)
            )
            gm_means = np.array(observations.get("test_means"))
            gm_weights = np.array(observations.get("test_weights"))
            gm_cov = np.array(observations.get("test_covariances"))
            scale_mu = observations.get("mean_y_train")
            scale_std = observations.get("std_y_train")
            assert len(observations.get("pred")) == len(observations.get("test_assign"))
            predictions = np.array(observations.get("pred"))
            pred_assignments = np.argmax(observations.get("test_assign"), axis=1)
            true_observations = np.array(observations.get("trues"))
            true_assignments = np.array(true_observations > threshold, dtype=int)
            train_X = np.array(observations.get("train_X"))
            # train_Y = np.array(observations.get('train_Y'))
            zo_loss_per_split = zero_one_loss(true_assignments, pred_assignments)
            pred_comp0 = predictions[pred_assignments == 0]
            pred_comp1 = predictions[pred_assignments == 1]
            color_assignment = [mean_colors[x] for x in pred_assignments]
            ax[j, i].scatter(
                true_observations, predictions, color=color_assignment, alpha=0.7
            )
            ax[j, i].set_xlabel("true values")
            ax[j, i].set_ylabel("predicted values")
            # ax[j,i].hist([pred_comp1, pred_comp2], color=mean_colors, bins=100, label=["g1", "g2"])
            # ax[j,i].hist(true_observations, bins=150, alpha=0.5, label="truth")
            ax[j, i].vlines(
                threshold,
                ymin=0,
                ymax=2.0,
                colors="k",
                linestyles="dashdot",
                label=f"threshold",
            )
            ax[j, i].hlines(
                threshold,
                xmin=0,
                xmax=2.0,
                colors="k",
                linestyles="dashdot",
                label=f"threshold",
            )
            ax[j + 2, i].scatter(
                train_X[:, 0], train_X[:, 1], alpha=0.025, label=f"train"
            )
            for c in range(method.n_components):
                mean = (
                    gm_means[c][-1] * scale_std + scale_mu
                )  # unscaling, since GMM is fitted in scaled space
                sigma = np.sqrt(gm_cov[c][-1, -1] * scale_std)
                ax[j, i].plot(
                    mean, mean, c=marker_colors[c], marker="o", label=f"mu_{str(c)}"
                )
                ax[j, i].plot(
                    mean * gm_weights[c],
                    mean * gm_weights[c],
                    color=marker_colors[c],
                    marker="x",
                    label=f"w*mu_{str(c)}",
                )
                ax[j, i].plot(
                    [mean - sigma, mean + sigma],
                    [mean, mean],
                    color=marker_colors[c],
                    label="var(y)",
                )
                ax[j, i].plot(
                    [mean, mean],
                    [mean - sigma, mean + sigma],
                    color=marker_colors[c],
                    label="var(y)",
                )
                ax[j, i].plot(
                    [(mean - sigma) * gm_weights[c], (mean + sigma) * gm_weights[c]],
                    [mean * gm_weights[c], mean * gm_weights[c]],
                    color=marker_colors[c],
                    linestyle="dashed",
                    label="w*var(y)",
                )
                xx = np.linspace(mean - 2 * sigma, mean + 2 * sigma, 100)
                performance_g0 = (
                    mean_squared_error(
                        true_observations[pred_assignments == 0], pred_comp0
                    )
                    if len(pred_comp0) > 0
                    else -1
                )
                performance_g1 = (
                    mean_squared_error(
                        true_observations[pred_assignments == 1], pred_comp1
                    )
                    if len(pred_comp1) > 0
                    else -1
                )
                ax[j + 1, i].plot([0, 1], [performance_g0, performance_g1], "ko")
                ax[j + 2, i].scatter(
                    gm_means[c][0], gm_means[c][1], s=100, c="red", marker="x", label=c
                )
            ax[j, i].set_title(
                f"{rep}@d={d} \n zero-one loss: {np.round(np.mean(zo_loss_per_split), 3)}"
            )
            ax[j + 2, i].set_xlabel("PC1")
            ax[j, i].set_xlabel("observations")
            if i == 0:
                ax[j + 2, i].set_ylabel("PC2")
                ax[j, i].set_ylabel("count")
    ax[j, i].legend()
    ax[j + 1, i].legend()
    plt.suptitle(f"Mixture Visualization \n {dataset[0]} {protocol}")
    plt.tight_layout()
    plt.savefig(
        f"./results/figures/mixtures/{dataset}_{protocol}_gmm_visualization.png",
        bbox_inches="tight",
    )
    plt.savefig(
        f"./results/figures/mixtures/{dataset}_{protocol}_gmm_visualization.pdf",
        bbox_inches="tight",
    )
    plt.show()


def __filter_results_given_threshold(
    results: dict, t: float, metrics: list, observation_id="trues"
) -> dict:
    """
    Filter results dict by observation_id values over CV entries against threshold value t.
    Return dict with observation and metric values.
    """
    filtered_results = {}
    for split in results.keys():
        filtered_results[split] = {}
        idx = np.where(np.array(results.get(split).get(observation_id)) >= t)[0]
        filtered_results[split][observation_id] = np.array(
            results[split].get(observation_id)
        )[idx]
        for metric in metrics:
            filtered_results[split][metric] = np.array(results[split].get(metric))[idx]
    return filtered_results


def __compute_metric_from_filtered_results(
    filtered_results_array: list, metric_name: str
) -> list:
    result_metric = []
    for results in filtered_results_array:
        metric_list = []
        for cv_split in results.keys():
            y_truth = np.array(results.get(cv_split).get("trues"))
            y_pred = np.array(results.get(cv_split).get("pred"))
            assert len(y_truth) == len(y_pred)
            if not bool(any(y_truth)) and not bool(any(y_pred)):
                continue
            if metric_name == MSE:
                metric = mean_squared_error(y_truth, y_pred)
            elif metric_name == SPEARMAN_RHO:
                metric = spearmanr(y_truth, y_pred)
            else:
                raise NotImplementedError(f"Provided metric {metric} does not exist!")
            metric_list.append(metric)
        result_metric.append(np.mean(metric))
    return result_metric


def plot_metric_against_threshold(results: dict, metrics: list, protocol_name: str):
    dataset = list(results.keys())[0]
    methods = list(results.get(dataset).keys())
    representations = list(results.get(dataset).get(methods[0]).keys())
    fig, ax = plt.subplots(
        len(metrics), len(representations), figsize=(25, 15)
    )  # for each representation a column
    _results = results.get(dataset).get(methods[0]).get(representations[0]).get(None)
    cmap = plt.get_cmap("prism", 20)
    for i, metric in enumerate(metrics):
        for j, representation in enumerate(representations):
            for k, method in enumerate(methods):
                _results = (
                    results.get(dataset).get(method).get(representation).get(None)
                )
                observed_min = min(
                    [min(_results.get(i_cv).get("trues")) for i_cv in _results]
                )  # GET MIN MAX FROM OBSERVATIONS
                observed_max = max(
                    [max(_results.get(i_cv).get("trues")) for i_cv in _results]
                )
                observation_space = np.linspace(observed_min, observed_max, 100)
                n_splits = len(_results.keys())
                reported_metric = np.mean(
                    [_results.get(i).get(metric) for i in range(n_splits)]
                )
                ax[i, j].hlines(
                    y=reported_metric,
                    xmin=observed_min,
                    xmax=observed_max,
                    linestyles="dashed",
                    colors=cmap(k),
                    label=f"mean {metric} {method}",
                )
                filtered_observations = [
                    __filter_results_given_threshold(
                        results=_results, t=t, metrics=["pred", "mse"]
                    )
                    for t in observation_space
                ]
                n_results = [
                    len(result.get(0).get("trues")) for result in filtered_observations
                ]
                # filtered_metric = __compute_metric_from_filtered_results(filtered_observations, metric_name=metric)
                if metric == MSE:
                    filtered_metric = [
                        np.mean(
                            [
                                np.mean(split_result.get(metric.lower()))
                                for split_result in filtered_result.values()
                            ]
                        )
                        for filtered_result in filtered_observations
                    ]
                elif metric == SPEARMAN_RHO:
                    filtered_metric = [
                        np.mean(
                            [
                                spearmanr(
                                    split_result.get("trues"), split_result.get("pred")
                                )
                                for split_result in filtered_result.values()
                            ]
                        )
                        for filtered_result in filtered_observations
                    ]
                else:
                    raise NotImplementedError(
                        f"Specified metric {metric} not implemented!"
                    )
                ax[i, j].plot(
                    observation_space, filtered_metric, c=cmap(k), label=method
                )
            metric_name = metric if metric != MSE else "N" + MSE
            ax[i, j].set_ylabel(metric_name)
            if metric == MSE:
                ax[i, j].set_ylim((0.0, 1.5))
            else:
                ax[i, j].set_ylim((0.0, 1.0))
            ax[i, j].set_xticks(observation_space[::10])
            labels = [
                f"{np.round(t,1)} \n {n}"
                for t, n in zip(observation_space[::10], n_results[::10])
            ]
            ax[i, j].set_xticklabels(labels)
            ax[i, j].set_xlabel(f"observation threshold t \n N observations")
            ax[i, j].set_title(representation)
    plt.legend()
    plt.suptitle(f"Metric against observation threshold \n {protocol_name}")
    plt.savefig(
        f"./results/figures/relative_threshold/{dataset}_{protocol_name}_{metrics}_against_threshold.png"
    )
    plt.show()
