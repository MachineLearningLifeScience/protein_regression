import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from data.train_test_split import BlockPostionSplitter, RandomSplitter
from algorithms.gp_on_real_space import GPonRealSpace
from algorithms.one_hot_gp import GPOneHotSequenceSpace
from algorithms.random_forest import RandomForest
from algorithms.KNN import KNN
from gpflow.kernels import SquaredExponential
from sklearn.calibration import CalibrationDisplay

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.cm.get_cmap("Dark2")
algo_list = [GPonRealSpace().get_name(), GPonRealSpace(kernel=SquaredExponential()).get_name(), RandomForest().get_name(), KNN().get_name()]

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}

# TODO: load results from mlflow runs
for i, (clf, name) in enumerate(algo_list):
    display = CalibrationDisplay.from_predictions(
        y_true=...,
        y_prob=...,
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
    )
    calibration_displays[name] = display

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots (Naive Bayes)")

# Add histogram
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
for i, (_, name) in enumerate(algo_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()