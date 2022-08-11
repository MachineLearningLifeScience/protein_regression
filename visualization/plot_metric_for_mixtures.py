import numpy as np
from scipy import stats
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt
from algorithms import GMMRegression


def plot_metric_for_mixtures(results: dict, threshold:float, protocol: str, method=GMMRegression(), augmentation=None, split=0) -> None:
    """
    Plot mixture models results as multiplot (representations[y, X] x d_PCA)
    Mark Gaussians by color with mean and standard deviation for each.
    Compute Loss respective measurement threshold: cluster = bool(y > threshold)
    """
    # color by cluster
    # plot two rows: observations clustered, representations clustered, columns are dimensions (d=2,10,100,1000)
    dimensions = list(results.keys())
    dataset = list(results.get(dimensions[0]).keys())
    representations = list(results.get(dimensions[0]).get(dataset[0]).get(method.get_name()).keys())
    fig, ax = plt.subplots(2*len(representations), len(dimensions), figsize=(24, 15)) # for each representation two rows
    mean_colors = ["purple", "darkorange"]
    for i, d in enumerate(dimensions):
        for j, rep in enumerate(representations):
            j *= 2 # in steps of two
            observations = results.get(d).get(dataset[0]).get(method.get_name()).get(rep).get(augmentation).get(split)
            gm_means = np.array(observations.get('means'))
            gm_weights = np.array(observations.get('weights'))
            gm_cov = np.array(observations.get('covariances'))
            scale_mu = observations.get('mean_y_train')
            scale_std = observations.get('std_y_train')
            assert len(observations.get('pred')) == len(observations.get('test_assign'))
            predictions = np.array(observations.get('pred'))
            pred_assignments = np.argmax(observations.get('test_assign'), axis=1)
            true_observations = np.array(observations.get('trues'))
            true_assignments = np.array(true_observations>threshold, dtype=int)
            train_X = np.array(observations.get('train_X'))
            # train_Y = np.array(observations.get('train_Y'))
            zo_loss_per_split = zero_one_loss(true_assignments, pred_assignments)
            pred_comp1 = predictions[pred_assignments==0]
            pred_comp2 = predictions[pred_assignments==1]
            ax[j,i].hist([pred_comp1, pred_comp2], color=mean_colors, bins=100, label=["g1", "g2"])
            ax[j,i].hist(true_observations, bins=150, label="truth")
            ax[j,i].vlines(threshold, ymin=0, ymax=50, colors="k", linestyles="dashdot", label=f"threshold")
            ax[j+1,i].scatter(train_X[:, 0], train_X[:, 1], alpha=0.025, label=f"train")
            # ax[j+1,i].scatter(train_X[:, 0], train_X[:, 1], alpha=0.125, c=train_Y, label=f"train")
            for c in range(method.n_components):
                mean = gm_means[c][-1]*scale_std + scale_mu # unscaling, since GMM is fitted in scaled space
                sigma = np.sqrt(gm_cov[c][-1,-1]*scale_std) 
                ax[j,i].vlines(mean, ymin=0, ymax=50, colors=mean_colors[c], label=f"mu_{str(c)}")
                ax[j,i].vlines(mean*gm_weights[c], ymin=0, ymax=50, colors=mean_colors[c], linestyles="dotted", label=f"w*mu_{str(c)}")
                ax[j,i].hlines(10, xmin=mean-sigma, xmax=mean+sigma, colors=mean_colors[c], label="var(y)")
                ax[j,i].hlines(15, xmin=(mean-sigma)*gm_weights[c], 
                                    xmax=(mean+sigma)*gm_weights[c], colors=mean_colors[c], linestyles="dashed", label="w*var(y)")
                xx = np.linspace(mean-2*sigma, mean+2*sigma, 100)
                #ax[j,i].plot(xx, stats.norm.pdf(xx, mean, sigma), label=str(c))
                ax[j+1,i].scatter(gm_means[c][0], gm_means[c][1], s=100, c="red", marker="x", label=c)
            ax[j,i].set_title(f"{rep}@d={d} \n zero-one loss: {np.round(np.mean(zo_loss_per_split), 3)}")
            ax[j+1, i].set_xlabel("PC1")
            ax[j,i].set_xlabel("observations")
            if i==0:
                ax[j+1, i].set_ylabel("PC2")
                ax[j,i].set_ylabel("count")  
    plt.suptitle(f"Mixture Visualization \n {dataset[0]}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./results/figures/mixtures/{dataset}_{protocol}_gmm_visualization.png")
    plt.savefig(f"./results/figures/mixtures/{dataset}_{protocol}_gmm_visualization.pdf")
    plt.show()

