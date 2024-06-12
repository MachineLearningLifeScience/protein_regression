import numpy as np
from scipy.spatial.distance import euclidean
from data.train_test_split import WeightedTaskSplitter


np.random.seed(42)
X_s = np.array(np.linspace(0, 1, 30))[:, np.newaxis]
y_s = np.array(np.random.sample(X_s.shape[0]))[:, np.newaxis]
subset_size = 5
X_p = X_s[:subset_size] - 0.01
y_p = y_s[:subset_size] - 0.01
alphas = np.ones(X_s.shape[0]) * 0.1
eta = 0.125


def weight_REFERENCE(alphas, x_p, y_p, X_s, Y_s, eta) -> np.ndarray:
    """
    Reference weight computation function see Eq.(8) in J.Garcke, T.Vanck. Importance Weighted Inductive Transfer Learning for Regression. ECML PKDD 2014.
    """
    assert (
        alphas.shape[0] == X_s.shape[0] == Y_s.shape[0]
    ), "Alpha required with same dimensions as X_s and Y_s!"
    N = X_s.shape[0]
    w_vec = []
    for j in range(N):  # compute euclidean distance against each X_sY_s row
        x_p_y_p, x_s_y_s = np.hstack([x_p, y_p]), np.hstack([X_s[j], Y_s[j]])
        w_j = alphas[j] * np.exp(-euclidean(x_p_y_p, x_s_y_s) / (2 * (eta**2)))
        w_vec.append(w_j)
    return w_vec


def test_weight_shape():
    w = WeightedTaskSplitter.weight(alphas, X_p[0], y_p[0], X_s, y_s, eta=eta)
    assert w.shape == (1, X_s.shape[0])


def test_weight_reference_against_matrix_computation():
    M = X_p.shape[0]
    N = X_s.shape[0]
    ws = []
    ws_ref = []
    for i in range(M):
        w_ref = weight_REFERENCE(alphas, X_p[i], y_p[i], X_s, y_s, eta=eta)
        ws_ref.append(w_ref)
        w = WeightedTaskSplitter.weight(alphas, X_p[i], y_p[i], X_s, y_s, eta=eta)
        ws.append(w)
        assert np.array(w_ref)[np.newaxis, :].shape == np.array(w).shape
    ws_ref = np.array(ws_ref).reshape(M, N)
    ws = np.array(ws)[:, 0].reshape(
        M, N
    )  # initial euclidean distances account for first column of distance matrix!
    # np.testing.assert_equal(ws_ref, ws)
    np.testing.assert_almost_equal(ws_ref, ws, decimal=12)


def average_log_weight_REFERENCE(alphas, X_s, Y_s, X_p, Y_p) -> float:
    """
    Reference log weight computation iteratively over X_pY_p set.
    See Eq.(16) in J.Garcke 2014.
    """
    assert (
        alphas.shape[0] == X_s.shape[0] == Y_s.shape[0]
    ), "Alpha required with same dimensions as X_s and Y_s!"
    M = X_p.shape[0]
    ws = []
    for i in range(M):
        _w = weight_REFERENCE(
            alphas=alphas, x_p=X_p[i], y_p=Y_p[i], X_s=X_s, Y_s=Y_s, eta=eta
        )
        ws.append(_w)
    w = np.sum(np.log(ws)) / M
    return -w  # negative as we minimize


def test_weight_avrg_reference_against_vectorized():
    w_ref = average_log_weight_REFERENCE(alphas, X_s, y_s, X_p, y_p)
    wts = WeightedTaskSplitter("test")
    wts.X_s = X_s
    wts.Y_s = y_s
    wts.eta = eta
    w = wts.average_log_weight(alphas, X_p=X_p, Y_p=y_p)
    np.testing.assert_approx_equal(w_ref, w)  # randomness introduces slight divergences


def test_optimized_N_constraint():
    wts = WeightedTaskSplitter("test")
    wts.X_s = X_s
    wts.Y_s = y_s
    alphas = wts._optimize(X_p, y_p)
    # constraint applies to computation with X_sY_s
    constraint = wts.constrained_weight_sum(alphas)
    print(constraint)
    np.testing.assert_almost_equal(0.0, constraint)
