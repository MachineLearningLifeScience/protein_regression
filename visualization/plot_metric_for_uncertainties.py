import os
import pickle
from os.path import join
from typing import List

import imageio
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import Polygon

from uncertainty_quantification.calibration import prep_reliability_diagram
from uncertainty_quantification.chi_squared import chi_squared_anees
from uncertainty_quantification.confidence import quantile_and_oracle_errors
from util.mlflow.constants import (
    ESM,
    ESM1V,
    ESM2,
    EVE,
    GP_L_VAR,
    MSE,
    NO_AUGMENT,
    OBSERVED_Y,
    ONE_HOT,
    PROTT5,
    PSSM,
    RF_ESTIMATORS,
    STD_Y,
    TRANSFORMER,
)
from util.mlflow.convenience_functions import get_mlflow_results_artifacts
from visualization import algorithm_colors as ac
from visualization import algorithm_markers as am

# MLFLOW CODE ONLY WORKS WITH THE BELOW LINE:
mlflow.set_tracking_uri("file:" + join(os.getcwd(), join("results", "mlruns")))


def plot_confidence_curve(
    h_i: np.ndarray, h_o: np.ndarray, savefig=True, suffix="", metric=MSE
) -> None:
    quantiles = np.arange(0, 1, 1 / len(h_i))
    plt.scatter(quantiles, h_i, "r")
    plt.plot(quantiles, h_i, "r-", label="predictions")
    plt.scatter(quantiles, h_o, "k")
    plt.plot(quantiles, h_o, "k-", label="oracle")
    plt.ylabel(f"{metric}")
    plt.xlabel("quantile")
    plt.title("Confidence Curve \n quantile ranked loss")
    # TODO: add AUCO
    plt.legend()
    # TODO: savefig
    plt.show()


def plot_calibration(fractions, savefig=True, suffix="") -> None:
    quantiles = np.arange(0, 1, 1 / len(fractions))
    plt.scatter(quantiles, fractions, "r")
    plt.plot(quantiles, fractions, "r-", label="calibration")
    plt.plot(np.arange(0, 1), "k:", alpha=0.5)
    plt.legend()
    plt.show()


def combine_pointsets(x1, x2):
    """
    AUTHOR: JKH
    Function takes two lists and combines them to a list that is
    ready to be fed to shapely.Polygon class by outputting a new list
    with each element being a point on the polygon. This is
    implemented by walking along x1 and then walking on x2
    and concat first point of x1. This assumes that x1 and x2 are
    on same x-axis and each element in x1 and x2 matches on the x-axis
    with x-axis [0,1] length == len(x1) == len(x2)

    Input:
      x1: list-type of values
      x2: list-type of values
    Output:
      Shapely Polygon ready list
    """
    x0 = np.flip(np.linspace(0, 1, len(x1)))
    p1 = list(zip(x0, x1))[::-1]
    p2 = list(zip(x0, x2))
    return p1 + p2 + [p1[0]]


def plot_polygon(ax, poly, **kwargs):
    """
    AUTHOR: Jacob KH
    Input:
      ax: matplotlib.axes, e.g fig, ax = plt.subplots()
      pgon: shapely Polygon
    Output:
      plots the input polygon
    Note: "pip install shapely"
    """
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors],
    )

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def chi_square_fig(
    metric_values: dict,
    cvtype: str = "",
    dataset="",
    representation="",
    optimize_flag=False,
    dim=None,
    dim_reduction=None,
    savefig=True,
):
    """
    Visualize X**2 statistic on predictions.
    AUTHOR: RM
    """
    filename = f"results/figures/uncertainties/Xi_squared_{cvtype}_{dataset}_{representation}_opt_{optimize_flag}_d_{dim}{dim_reduction}"
    algos = []
    name_dict = {
        ONE_HOT: "One-Hot",
        PSSM: "PSSM",
        EVE: "EVE",
        TRANSFORMER: "ProtBert",
        ESM: "ESM-1b",
        ESM1V: "ESM-1v",
        ESM2: "ESM2",
        PROTT5: "ProtT5",
    }
    font_kwargs = {"family": "Arial", "fontsize": 30, "weight": "bold"}
    font_kwargs_small = {"family": "Arial", "fontsize": 16, "weight": "bold"}
    n_prot = len(metric_values.keys())
    for d in metric_values.keys():
        for a in metric_values[d].keys():
            n_reps = len(metric_values[d][a].keys())
    fig, axs = plt.subplots(
        n_prot,
        n_reps,
        figsize=(4 * n_reps, 4 * n_prot + 1),
        squeeze=False,
        sharey="row",
    )  # gridspec_kw={'height_ratios': [4, 1]})
    algos = []
    for d, dataset_key in enumerate(metric_values.keys()):
        for i, algo in enumerate(metric_values[dataset_key].keys()):
            if algo not in algos:
                algos.append(algo)
            for j, rep in enumerate(metric_values[dataset_key][algo].keys()):
                for aug in metric_values[dataset_key][algo][rep].keys():
                    number_splits = len(
                        list(metric_values[dataset_key][algo][rep][aug].keys())
                    )
                    if algo not in algos:
                        algos.append(algo)
                    plt_idx = (d, j)
                    if j == 0:  # first column annotate y-axis
                        axs[plt_idx].set_ylabel(r"$\mathbf{\chi^2}$", **font_kwargs)
                        axs[plt_idx].yaxis.set_label_coords(
                            -0.225, 0.5, transform=axs[plt_idx].transAxes
                        )
                    chis = []
                    for s in metric_values[dataset_key][algo][rep][aug].keys():
                        pred_var = np.array(
                            metric_values[dataset_key][algo][rep][aug][s]["unc"]
                        )
                        data_uncertainties = metric_values[dataset_key][algo][rep][aug][
                            s
                        ].get(GP_L_VAR)
                        if data_uncertainties:  # in case of GP include data-noise
                            _scale_std = metric_values[dataset_key][algo][rep][aug][
                                s
                            ].get(STD_Y)
                            pred_var += np.sqrt(data_uncertainties * _scale_std)
                        y_true = np.array(
                            metric_values[dataset_key][algo][rep][aug][s]["trues"]
                        )
                        y_pred = np.array(
                            metric_values[dataset_key][algo][rep][aug][s]["pred"]
                        )
                        standardized_chi = chi_squared_anees(
                            y_true, y_pred, pred_var
                        ) / (
                            len(y_true) - 1
                        )  # NOTE: unbiased standardize: chi^2 * 1/(N-1)
                        if standardized_chi == np.nan:
                            print(f"NaN in Xi^2 for {dataset_key}, {algo}, {rep}")
                        chis.append(standardized_chi)
                    yerr = np.std(chis) / np.sqrt(
                        len(chis)
                    )  # NOTE: std. error across CV splits
                    # print(f"ch.squ. value: {np.mean(chis)}")
                    axs[plt_idx].errorbar(
                        i,
                        np.mean(chis),
                        yerr=yerr,
                        c=ac.get(algo),
                        marker="D",
                        fillstyle="full",
                        ms=12,
                        lw=2,
                        linestyle="-",
                        capsize=4.0,
                        label=algo,
                    )
                    if np.mean(chis) > 10e2:
                        axs[plt_idx].text(
                            i - 0.2,
                            65,
                            "{:.0f}".format(np.round(np.mean(chis), 0)),
                            color=ac.get(algo),
                            **font_kwargs_small,
                        )
                    axs[plt_idx].hlines(
                        1.0,
                        -0.26,
                        len(metric_values[dataset_key].keys()) - 0.0125,
                        linestyles="dotted",
                        color="k",
                        label="Perfect Calibration",
                        linewidths=2.5,
                    )
                    axs[plt_idx].fill_between(
                        np.arange(
                            -0.26, len(metric_values[dataset_key].keys()) + 0.0125
                        ),
                        0.5,
                        1.5,
                        alpha=0.0125,
                        color="k",
                    )
                    axs[plt_idx].set_title(name_dict[rep], **font_kwargs)
                    axs[plt_idx].set_yscale("log")
                    if dataset_key.lower() == "1fqg":
                        axs[plt_idx].set_ylim((10e-2, 10e1))
                    else:
                        axs[plt_idx].set_ylim((10e-4, 10e4))
                    axs[plt_idx].set_xlim(
                        (-0.24, float(len(metric_values[dataset_key].keys()) - 0.5))
                    )
                    axs[plt_idx].yaxis.set_tick_params(labelsize=20)
                    axs[plt_idx].grid(
                        True, color="k", alpha=0.125, which="both", linestyle="--"
                    )
                    axs[plt_idx].set_xticks([])
                    # align text in pairs of two around curves: upper left two, lower right two
    # plt.suptitle(f"{str(dataset)} Chi squ. (standardized) Stat. Split: {cvtype}", fontsize=22)
    plt.tight_layout()
    handles, labels = axs[plt_idx].get_legend_handles_labels()
    fig.legend(
        handles[(len(algos) - 1) :],
        labels[(len(algos) - 1) :],
        loc="lower center",
        ncol=len(algos) + 1,
        prop={"size": 19},
    )
    plt.subplots_adjust(wspace=0.1, left=0.08, right=0.975, bottom=0.2)
    if savefig:
        plt.savefig(filename + ".png")
        plt.savefig(filename + ".pdf")
    plt.show()


def reliabilitydiagram(
    metric_values: dict,
    number_quantiles: int,
    cvtype: str = "",
    dataset="",
    representation="",
    optimize_flag=False,
    dim=None,
    dim_reduction=None,
    savefig=True,
):
    """
    Plotting calibration Curves.
    AUTHOR: JKH,
    LAST CHANGES: RM
    """
    filename = f"results/figures/uncertainties/{cvtype}_reliabilitydiagram_{dataset}_{representation}_opt_{optimize_flag}_d_{dim}{dim_reduction}"
    algos = []
    n_prot = len(metric_values.keys())
    header_dict = {
        "1FQG": r"$\beta$-Lactamase",
        "UBQT": "Ubiquitin",
        "TOXI": "ParD-Antitoxin",
        "TIMB": "TIM-Barrel",
        "MTH3": "MTH3",
        "BRCA": "BRCA",
    }
    name_dict = {
        ONE_HOT: "One-Hot",
        PSSM: "PSSM",
        EVE: "EVE",
        TRANSFORMER: "ProtBert",
        ESM: "ESM-1b",
        ESM1V: "ESM-1v",
        ESM2: "ESM2",
        PROTT5: "ProtT5",
    }
    cv_names_dict = {
        "RandomSplitter": "Random CV",
        "PositionSplitter_p15": "Position CV",
        "BioSplitter1_2": r"1M $\rightarrow$ 2M",
        "BioSplitter2_2": r"2M $\rightarrow$ 2M",
        "BioSplitter2_3": r"2M $\rightarrow$ 3M",
        "BioSplitter3_3": r"3M $\rightarrow$ 3M",
        "BioSplitter3_4": r"3M $\rightarrow$ 4M",
    }
    font_kwargs = {"family": "Arial", "fontsize": 30, "weight": "bold"}
    font_kwargs_small = {"family": "Arial", "fontsize": 20, "weight": "bold"}
    for d in metric_values.keys():
        for a in metric_values[d].keys():
            n_reps = len(metric_values[d][a].keys())
    fig, axs = plt.subplots(
        n_prot * 2, n_reps, figsize=(3 * n_reps, 7 * n_prot), squeeze=False
    )  # gridspec_kw={'height_ratios': [4, 1]})
    algos = []
    for d, dataset_key in enumerate(metric_values.keys()):
        for i, algo in enumerate(metric_values[dataset_key].keys()):
            for j, rep in enumerate(metric_values[dataset_key][algo].keys()):
                for aug in metric_values[dataset_key][algo][rep].keys():
                    number_splits = len(
                        list(metric_values[dataset_key][algo][rep][aug].keys())
                    )
                    if algo not in algos:
                        algos.append(algo)
                    plt_idx = (d, j)
                    if j == 0:  # first column annotate y-axis
                        axs[plt_idx].set_ylabel("confidence", **font_kwargs_small)
                        axs[plt_idx].set_ylabel("confidence", **font_kwargs_small)
                    count = []
                    uncertainties_list = []  # point of investigation
                    nan_count = 0.0
                    for s in metric_values[dataset_key][algo][rep][aug].keys():
                        trues = metric_values[dataset_key][algo][rep][aug][s]["trues"]
                        preds = metric_values[dataset_key][algo][rep][aug][s]["pred"]
                        # report failing method by NaNs in predictions
                        nan_count += np.count_nonzero(np.isnan(preds))
                        uncertainties = np.sqrt(
                            metric_values[dataset_key][algo][rep][aug][s]["unc"]
                        )
                        data_uncertainties = metric_values[dataset_key][algo][rep][aug][
                            s
                        ].get(GP_L_VAR)
                        if data_uncertainties:  # in case of GP include data-noise
                            _scale_std = metric_values[dataset_key][algo][rep][aug][
                                s
                            ].get(STD_Y)
                            uncertainties += np.sqrt(data_uncertainties * _scale_std)
                        # confidence calibration
                        c, perc, e, s = prep_reliability_diagram(
                            trues, preds, uncertainties, number_quantiles
                        )
                        # C, perc = confidence_based_calibration(preds, uncertainties, y_ref_mean=np.mean(trues))
                        count.append(c)
                        uncertainties_list.append(np.array(uncertainties))
                    try:
                        count = np.mean(np.vstack(count), axis=0)
                        uncertainties = np.concatenate(uncertainties_list)
                    except ValueError:
                        print(f"Missing values for {algo} at {rep}.")
                        continue
                    axs[plt_idx].plot(
                        perc,
                        count,
                        c=ac.get(algo),
                        marker=am.get(algo),
                        fillstyle="none",
                        ms=8,
                        lw=2,
                        linestyle="-",
                        label=algo,
                    )
                    axs[plt_idx].plot(
                        perc,
                        perc,
                        ls=":",
                        color="k",
                    )  # label='Perfect Calibration'
                    axs[plt_idx].set_title(name_dict.get(rep), **font_kwargs)
                    axs[plt_idx].yaxis.set_tick_params(labelsize=18)
                    axs[plt_idx].xaxis.set_tick_params(labelsize=18)
                    axs[d + 1, j].hist(
                        uncertainties,
                        100,
                        label=f"{algo}; {rep}",
                        alpha=0.7,
                        color=ac.get(algo),
                    )
                    axs[plt_idx].set_xlabel("percentile", **font_kwargs_small)
                    axs[d + 1, j].set_xlabel("pred. unc.", **font_kwargs_small)
                    axs[d + 1, j].yaxis.set_tick_params(labelsize=18)
                    axs[d + 1, j].xaxis.set_tick_params(labelsize=18)
                    axs[plt_idx].set_xlim(0, 1.0)
    handles, labels = axs[plt_idx].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(algos) + 1, prop={"size": 17}
    )
    plt.subplots_adjust(
        wspace=0.28, hspace=0.33, left=0.05, right=0.97, top=0.855, bottom=0.175
    )
    plt.suptitle(
        f"{header_dict.get(dataset[0])} {cv_names_dict.get(cvtype)}", **font_kwargs
    )
    plt.xticks(size=20)
    plt.yticks(size=20)
    if savefig:
        plt.savefig(filename + ".png")
        plt.savefig(filename + ".pdf")
    plt.show()


def multi_dim_reliabilitydiagram(
    metric_values: dict,
    number_quantiles: int,
    cvtype: str = "",
    dataset="",
    representation="",
    optimize_flag=True,
    dim_reduction=None,
):
    """
    Plotting calibration Curves including results from lower dimensions.
    AUTHOR: RM with utility functions JKH
    """
    colors = ac.values()
    markers = [
        plt.Line2D([0, 0], [0, 0], color=color, marker="o", linestyle="")
        for color in colors
    ]
    algos = []
    dim = list(metric_values.keys())[0]
    data = list(metric_values[dim].keys())[0]
    alg = list(metric_values[dim][data].keys())[0]
    n_algs = len(metric_values[dim][data].keys())
    n_reps = len(metric_values[dim][data][alg].keys())
    fig, axs = plt.subplots(
        n_algs * 2,
        n_reps,
        figsize=(21, 25),
        gridspec_kw={"height_ratios": [4, 1] * n_algs},
    )
    dimensions = list(metric_values.keys())[:-1] + [1128]
    shade = np.arange(0.2, 1.1, step=1 / len(dimensions))
    for d_idx, (dim, _results) in enumerate(metric_values.items()):
        for d, dataset_key in enumerate(_results.keys()):
            row_idx = 0
            for i, algo in enumerate(_results[dataset_key].keys()):
                for j, rep in enumerate(_results[dataset_key][algo].keys()):
                    for aug in _results[dataset_key][algo][rep].keys():
                        if f"d={str(dim)} {str(algo)}" not in algos:
                            algos.append(f"d={str(dim)} {str(algo)}")
                        count = []
                        uncertainties_list = []  # point of investigation
                        for s in _results[dataset_key][algo][rep][aug].keys():
                            trues = _results[dataset_key][algo][rep][aug][s]["trues"]
                            preds = _results[dataset_key][algo][rep][aug][s]["pred"]
                            uncertainties = np.sqrt(
                                _results[dataset_key][algo][rep][aug][s]["unc"]
                            )
                            data_uncertainties = _results[dataset_key][algo][rep][aug][
                                s
                            ].get(GP_L_VAR)
                            if data_uncertainties:  # in case of GP include data-noise
                                scaling_std = _results[dataset_key][algo][rep][aug][
                                    s
                                ].get(STD_Y)
                                uncertainties += np.sqrt(
                                    data_uncertainties * scaling_std
                                )
                            # confidence calibration
                            C, perc, E, S = prep_reliability_diagram(
                                trues, preds, uncertainties, number_quantiles
                            )
                            count.append(C)
                            uncertainties_list.append(np.array(uncertainties))
                        count = np.mean(np.vstack(count), axis=0)
                        uncertainties = np.concatenate(uncertainties_list)
                        axs[row_idx, j].plot(
                            perc,
                            count,
                            c=ac.get(algo),
                            lw=2,
                            linestyle="-",
                            alpha=shade[d_idx],
                        )
                        axs[row_idx, j].plot(
                            perc,
                            count,
                            color=ac.get(algo),
                            marker="o",
                            alpha=shade[d_idx],
                            markersize=5 + 2 * shade[d_idx],
                            label=f"d={str(dim)} {str(algo)}",
                        )
                        if d_idx == 0:
                            axs[row_idx, j].plot(
                                perc,
                                perc,
                                ls=":",
                                color="k",
                                label="Perfect Calibration",
                            )
                            axs[row_idx, 0].set_ylabel("cm. confidence", size=9)
                            axs[row_idx + 1, 0].set_ylabel("count", size=9)
                        axs[row_idx, j].set_title(f"{algo} on {rep}")
                        axs[row_idx + 1, j].hist(
                            uncertainties,
                            100,
                            label=f"{rep} d={dim}",
                            alpha=shade[d_idx],
                            color=ac.get(algo),
                        )
                        axs[row_idx, j].set_xlabel("percentile", size=12)
                        axs[row_idx + 1, j].set_xlabel("std", size=12)
                        if d_idx + 1 == len(metric_values.keys()):
                            axs[row_idx, j].legend(loc="lower right", prop={"size": 4})
                row_idx += 2
    plt.suptitle(f"{str(dataset)} Calibration Split: {cvtype}, {dim_reduction}")
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.tight_layout()
    filename = f"results/figures/uncertainties/dim_{str(list(metric_values.keys()))}_{cvtype}_reliabilitydiagram_{dataset}_{representation}_opt_{optimize_flag}_d_{str(dimensions)}{dim_reduction}"
    plt.savefig(filename + ".png")
    plt.savefig(filename + ".pdf")
    plt.show()


def confidence_curve(
    metric_values: dict,
    number_quantiles: int,
    cvtype: str = "",
    dataset="",
    representation="",
    optimize_flag=True,
    dim=None,
    dim_reduction=None,
    savefig=True,
):
    qs = np.linspace(0, 1, 1 + number_quantiles)
    n_prot = len(list(metric_values.keys()))
    for d in metric_values.keys():
        n_algo = len(metric_values[d].keys())
        for a in metric_values[d].keys():
            n_reps = len(metric_values[d][a].keys())
    fig, axs = plt.subplots(n_prot, n_reps, figsize=(20, 17.5))
    for d, dataset_key in enumerate(metric_values.keys()):
        algos = []
        for i, algo in enumerate(metric_values[dataset_key].keys()):
            for j, rep in enumerate(metric_values[dataset_key][algo].keys()):
                for aug in metric_values[dataset_key][algo][rep].keys():
                    if algo not in algos:
                        algos.append(algo)
                    quantile_errs, oracle_errs = [], []
                    for s in metric_values[dataset_key][algo][rep][aug].keys():
                        uncertainties = np.sqrt(
                            metric_values[dataset_key][algo][rep][aug][s]["unc"]
                        )
                        data_uncertainties = metric_values[dataset_key][algo][rep][aug][
                            s
                        ].get(GP_L_VAR)
                        if (
                            data_uncertainties
                        ):  # in case of GP include data-noise unscaled
                            scale_std = metric_values[dataset_key][algo][rep][aug][
                                s
                            ].get(STD_Y)
                            uncertainties += np.sqrt(data_uncertainties * scale_std)
                        errors = metric_values[dataset_key][algo][rep][aug][s]["mse"]
                        # Ranking-based calibration
                        qe, oe = quantile_and_oracle_errors(
                            uncertainties, errors, number_quantiles
                        )
                        quantile_errs.append(qe)
                        oracle_errs.append(oe)
                    quantile_errs = np.mean(np.vstack(quantile_errs), 0)
                    oracle_errs = np.mean(np.vstack(oracle_errs), 0)
                    try:
                        pgon = Polygon(
                            combine_pointsets(quantile_errs, oracle_errs)
                        )  # Assuming the OP's x,y coordinates
                    except:
                        print("ERROR building Polygon with")
                        print(f"Quantile: {quantile_errs}")
                        print(f"Oracle: {oracle_errs}")
                    # plot_polygon(axs[d,j], pgon, facecolor='red', edgecolor='red', alpha=0.12)
                    axs[d, j].plot(
                        qs,
                        np.flip(quantile_errs),
                        c=ac.get(algo),
                        marker=am.get(algo),
                        fillstyle="none",
                        ms=8,
                        lw=2,
                        linestyle="-",
                        label=algo,
                    )
                    # axs[d,j].text(0, 0-i, f'AUCO {algo}: {np.round(pgon.area,3)}\nError drop: {np.round(quantile_errs[-1]/quantile_errs[0],3)}', transform=axs[d,j].transAxes)
                    axs[d, j].plot(
                        qs, np.flip(oracle_errs), "k--", lw=2, label="Oracle"
                    )
                    axs[d, j].set_ylabel("NMSE", size=19)
                    axs[d, j].set_title(
                        f"Confidence Curves: {dataset_key} Algo: {algo}  Rep: {rep}",
                        size=20,
                    )
                axs[-1, j].set_xlabel("Percentile", size=19)
    filename = f"results/figures/uncertainties/{algo}_{cvtype}_confidence_curve_{dataset}_{representation}_opt_{optimize_flag}_d_{dim}_{dim_reduction}"
    plt.legend(prop={"size": 19})
    plt.suptitle(f"{str(dataset)} Split: {cvtype} , d={dim} {dim_reduction}")
    # plt.tight_layout()
    if savefig:
        plt.savefig(filename + ".png")
        plt.savefig(filename + ".pdf")
    plt.show()


def multi_dim_confidencecurve(
    metric_values: dict,
    number_quantiles: int,
    cvtype: str = "",
    dataset="",
    representation="",
    optimize_flag=True,
    dim_reduction=None,
):
    dimensions = list(metric_values.keys())[:-1] + [1128]
    qs = np.linspace(0, 1, 1 + number_quantiles)
    dim = list(metric_values.keys())[0]
    data = list(metric_values[dim].keys())[0]
    rep = list(metric_values[dim][data].keys())[0]
    n_algo = len(metric_values[dim][data].keys())
    n_reps = len(metric_values[dim][data][rep].keys())
    fig, axs = plt.subplots(n_algo, n_reps, figsize=(15, 15))
    shade = np.arange(0.2, 1.1, step=1 / len(dimensions))
    for d_idx, (dim, _results) in enumerate(metric_values.items()):
        for d, dataset_key in enumerate(_results.keys()):
            for i, algo in enumerate(_results[dataset_key].keys()):
                for j, rep in enumerate(_results[dataset_key][algo].keys()):
                    for aug in _results[dataset_key][algo][rep].keys():
                        quantile_errs, oracle_errs = [], []
                        for s in _results[dataset_key][algo][rep][aug].keys():
                            uncertainties = np.sqrt(
                                _results[dataset_key][algo][rep][aug][s]["unc"]
                            )
                            gp_data_uncertainties = _results[dataset_key][algo][rep][
                                aug
                            ][s].get(GP_L_VAR)
                            if (
                                gp_data_uncertainties
                            ):  # in case of GP include data-noise unscaled
                                scale_std = _results[dataset_key][algo][rep][aug][
                                    s
                                ].get(STD_Y)
                                uncertainties += np.sqrt(
                                    gp_data_uncertainties * scale_std
                                )
                            errors = _results[dataset_key][algo][rep][aug][s]["mse"]
                            # Ranking-based calibration
                            qe, oe = quantile_and_oracle_errors(
                                uncertainties, errors, number_quantiles
                            )
                            quantile_errs.append(qe)
                            oracle_errs.append(oe)
                        quantile_errs = np.mean(np.vstack(quantile_errs), 0)
                        oracle_errs = np.mean(np.vstack(oracle_errs), 0)
                        axs[i, j].plot(
                            qs,
                            np.flip(quantile_errs),
                            lw=2,
                            alpha=shade[d_idx],
                            color="k",
                            label=f"d={dim}",
                        )
                        if d_idx == 0:  # TODO investigate / correct oracle errors
                            axs[i, j].plot(
                                qs, np.flip(oracle_errs), "k--", lw=2, label="Oracle"
                            )
                        axs[i, j].set_ylabel("NMSE", size=11)
                        axs[i, j].set_xlabel("Percentile", size=11)
                        axs[i, j].set_title(f"Rep: {rep} Algo: {algo}", size=12)
                        axs[i, j].set_ylim([0, 1.75])
            axs[i, j].legend()
    plt.suptitle(f"{str(dataset)} Split: {cvtype} ; {dim_reduction}")
    plt.tight_layout()
    filename = f"results/figures/uncertainties/{algo}_{cvtype}_confidence_curve_{dataset}_{representation}_opt_{optimize_flag}_d_{str(dimensions)}_{dim_reduction}"
    plt.savefig(filename + ".png", bbox_inches="tight")
    plt.savefig(filename + ".pdf", bbox_inches="tight")
    plt.show()


def plot_uncertainty_eval(
    datasets: List[str],
    reps: List[str],
    algos: List[str],
    train_test_splitter,
    augmentations: str = [NO_AUGMENT],
    number_quantiles: int = 10,
    optimize: bool = True,
    d: int = None,
    dim_reduction: str = None,
    metrics: str = [
        GP_L_VAR,
        STD_Y,
        RF_ESTIMATORS,
    ],
    cached_results: bool = False,
    confidence_plot: bool = False,
    savefig=True,
):
    filename = f"/Users/rcml/protein_regression/results/cache/results_calibration_comparison_d={'_'.join(datasets)}_a={'_'.join(algos)}_r={'_'.join(reps)}_m={'_'.join(metrics)}_s={train_test_splitter.get_name()}.pkl"
    if cached_results and os.path.exists(filename):
        print(f"Loading cached results: {filename}")
        with open(filename, "rb") as infile:
            results_dict = pickle.load(infile)
    else:
        results_dict = get_mlflow_results_artifacts(
            datasets=datasets,
            reps=reps,
            metrics=metrics,
            algos=algos,
            train_test_splitter=train_test_splitter,
            augmentation=augmentations,
            dim=d,
            dim_reduction=dim_reduction,
            optimize=optimize,
        )
        with open(filename, "wb") as outfile:
            pickle.dump(results_dict, outfile)
    reliabilitydiagram(
        results_dict,
        number_quantiles,
        cvtype=train_test_splitter.get_name(),
        dataset=datasets,
        representation=reps,
        optimize_flag=optimize,
        dim=d,
        dim_reduction=dim_reduction,
        savefig=savefig,
    )
    chi_square_fig(
        results_dict,
        cvtype=train_test_splitter.get_name(),
        dataset=datasets,
        representation=reps,
        optimize_flag=optimize,
        dim=d,
        dim_reduction=dim_reduction,
        savefig=savefig,
    )
    if confidence_plot:
        confidence_curve(
            results_dict,
            number_quantiles,
            cvtype=train_test_splitter.get_name(),
            dataset=datasets,
            representation=reps,
            optimize_flag=optimize,
            dim=d,
            dim_reduction=dim_reduction,
            savefig=savefig,
        )


def plot_uncertainty_eval_across_dimensions(
    datasets: List[str],
    reps: List[str],
    algos: List[str],
    train_test_splitter,
    dimensions: List[int],
    augmentation=[NO_AUGMENT],
    number_quantiles=10,
    optimize=True,
    dim_reduction=None,
    metrics=[GP_L_VAR, STD_Y],
):
    dim_results_dict = {}
    for d in dimensions:
        dim_results_dict[d] = get_mlflow_results_artifacts(
            datasets=datasets,
            reps=reps,
            metrics=metrics,
            train_test_splitter=train_test_splitter,
            algos=algos,
            augmentation=augmentation,
            dim=d,
            dim_reduction=dim_reduction,
            optimize=optimize,
        )
    multi_dim_confidencecurve(
        dim_results_dict,
        number_quantiles,
        cvtype=train_test_splitter.get_name(),
        dataset=datasets[-1],
        representation=reps[-1],
        optimize_flag=optimize,
        dim_reduction=dim_reduction,
    )
    multi_dim_reliabilitydiagram(
        dim_results_dict,
        number_quantiles=number_quantiles,
        cvtype=train_test_splitter.get_name(),
        dataset=datasets[-1],
        representation=reps[-1],
        optimize_flag=optimize,
        dim_reduction=dim_reduction,
    )


def plot_uncertainty_optimization(
    dataset: str,
    rep: str,
    seeds: List[int],
    algos: List[str],
    number_quantiles: int,
    min_obs_metrics: dict,
    regret_metrics=dict,
    optimize=False,
    stepsize=10,
    max_iterations=500,
):
    # Note: optimize is set to false, as mlflow query is different for optimization experiments vs. regression tasks, no optimize flag is set for optimization tasks
    gif_filename = f"results/figures/optim/gif/optimization_experiment_calibration_{dataset}_{rep}.gif"
    gif_files_list = []
    results_dict = {}
    for s in seeds:
        experiment_ids = [dataset + "_optimization"]
        results_dict[s] = get_mlflow_results_artifacts(
            datasets=[dataset],
            algos=algos,
            reps=[rep],
            seed=s,
            optimize=optimize,
            experiment_ids=experiment_ids,
            metrics=[OBSERVED_Y, GP_L_VAR, STD_Y],
            train_test_splitter=None,
        )
    _recorded_algos = list(results_dict[seeds[0]][dataset].keys())
    for val_step in range(int(max_iterations / stepsize) - 1):
        step = 1 + val_step * stepsize
        filename = f"results/figures/optim/gif/optimization_experiment_{dataset}_{rep}_{val_step}"
        fig, ax = plt.subplots(
            3, len(algos), figsize=(12, 7), gridspec_kw={"height_ratios": [4, 1, 2]}
        )
        for k, algo in enumerate(_recorded_algos):
            count_list = []
            uncertainties_list = []
            best_observed_list = []
            regret_list = []
            # get artifacts at stepsize, make calibration plot
            for s_idx, seed in enumerate(seeds):
                _results = results_dict[seed][dataset][algo][rep][None][val_step]
                trues = _results["trues"]
                preds = _results["pred"]
                uncertainties = np.sqrt(_results["unc"])
                gp_data_uncertainties = results_dict[seed][dataset][algo][rep][None][
                    val_step
                ].get(GP_L_VAR)
                if gp_data_uncertainties:  # in case of GP include data-noise unscaled
                    scale_std = results_dict[seed][dataset][algo][rep][None][
                        val_step
                    ].get(STD_Y)
                    uncertainties += np.sqrt(gp_data_uncertainties * scale_std)
                # confidence calibration
                C, perc, E, S = prep_reliability_diagram(
                    trues, preds, uncertainties, number_quantiles
                )
                # C, perc = confidence_based_calibration(preds, uncertainties, y_ref_mean=np.mean(trues))
                count_list.append(C)
                uncertainties_list.append(np.array(uncertainties))
                best_observed_list.append(
                    min_obs_metrics[dataset][algo][rep][s_idx][:step]
                )
                regret_list.append(
                    regret_metrics[dataset][algo][rep][s_idx][:step]
                )  # TODO
            count = np.mean(np.vstack(count_list), axis=0)
            std_count = np.std(np.vstack(count_list), axis=0)
            uncertainties = np.concatenate(uncertainties_list)
            ax[0, k].plot(perc, count, c=ac[k], lw=2, linestyle="-", label=f"{algo}")
            ax[0, k].errorbar(
                perc, count, std_count, linestyle=None, c=ac[k], marker="o", ms=7
            )
            ax[0, k].plot(perc, perc, ls=":", color="k", label="Perfect Calibration")
            ax[0, k].set_title(f"{algo}")
            ax[1, k].hist(uncertainties, 100, alpha=0.7, color=ac[k])
            ax[1, k].set_xlim([0.0, 1.6])
            # ax[1, k].set_ylim([0, 750])
            ax[1, k].set_ylim([0, 1700])
            ax[0, k].set_xlabel("percentile", size=12)
            ax[1, k].set_xlabel("std", size=12)
            mean_best_observed_values = np.mean(np.vstack(best_observed_list), axis=0)
            std_observed_values = np.std(best_observed_list, ddof=1, axis=0) / np.sqrt(
                mean_best_observed_values.shape[0]
            )
            mean_regret_values = np.mean(np.vstack(regret_list), axis=0)
            std_regret_values = np.std(regret_list, ddof=1, axis=0) / np.sqrt(
                std_observed_values.shape[0]
            )
            ax[2, 0].plot(
                mean_best_observed_values, color=ac[k], label=algo, linewidth=2
            )  # best observed values over steps
            ax[2, 0].fill_between(
                list(range(len(mean_best_observed_values))),
                mean_best_observed_values - std_observed_values,
                mean_best_observed_values + std_observed_values,
                color=ac[k],
                alpha=0.5,
            )
            ax[2, 1].plot(
                mean_regret_values, color=ac[k], label=algo, linewidth=2
            )  # regret over steps
            ax[2, 1].fill_between(
                list(range(len(mean_regret_values))),
                mean_regret_values - std_regret_values,
                mean_regret_values + std_regret_values,
                color=ac[k],
                alpha=0.5,
            )
        ax[2, 0].set_xlabel("Iterations", size=16)
        ax[2, 0].set_ylim([-0.41, 1.1])
        ax[2, 1].set_xlabel("Iterations", size=16)
        ax[0, 0].set_ylabel("confidence", size=9)
        ax[1, 0].set_ylabel("count", size=9)
        ax[2, 0].set_ylabel("observed value", size=9)
        ax[2, 0].set_title("Best observed value")
        ax[2, 1].set_title("Regret values")
        markers = [
            plt.Line2D([0, 0], [0, 0], color=color, marker="o", linestyle="")
            for color in ac
        ]
        ax[2, 1].legend(
            markers, algos, loc="lower right", numpoints=1, prop={"size": 12}
        )
        gif_files_list.append(filename)
        plt.suptitle(f"Calibration on {rep}\n @ step {step}")
        plt.tight_layout()
        plt.savefig(filename + ".png")
        plt.savefig(filename + ".pdf")
        plt.close()
    # make gif
    with imageio.get_writer(gif_filename, mode="I") as writer:
        for filename in gif_files_list:
            image = imageio.imread(filename + ".png")
            for _ in range(2):
                writer.append_data(image)
    keep_files_idx = [
        0,
        1,
        2,
        3,
        4,
        9,
        19,
        49,
        99,
    ]  # keep some calibration plots for later assessment
    for i, filename in enumerate(set(gif_files_list[:-1])):
        if i in keep_files_idx:
            continue
        os.remove(filename + ".png")
        os.remove(filename + ".pdf")
