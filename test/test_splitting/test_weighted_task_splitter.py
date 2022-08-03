import numpy as np
from scipy.spatial.distance import euclidean
from data.train_test_split import WeightedTaskSplitter


def naive_weight(alphas, x_p, y_p, X_s, Y_s, eta) -> np.ndarray:
    assert alphas.shape[0] == X_s.shape[0] == Y_s.shape[0] , "Alpha required with same dimensions as X_s and Y_s!"
    N = X_s.shape[0]
    w_vec = []
    for j in range(N):
        w_j = alphas[j] * np.exp(-euclidean(np.hstack([x_p, y_p]), np.hstack([X_s[j], Y_s[j]]))/(2*(eta**2)))
        w_vec.append(w_j)
    return w_vec

X_s = np.array(np.linspace(0,1,100))[:, np.newaxis]
y_s = np.array(np.random.sample(X_s.shape[0]))[:, np.newaxis]
subset_size = 10
X_p = X_s[:subset_size] - 0.01
y_p = y_s[:subset_size] - 0.01
alphas = np.ones(X_s.shape) * 0.1
eta = 0.125


def test_naive_weight_against_matrix_computation():
    M = X_p.shape[0]
    N = X_s.shape[0]
    ws = []
    ws_ref = []
    for i in range(M):
        w_ref = naive_weight(alphas, X_p[i], y_p[i], X_s, y_s, eta=eta)
        ws_ref.append(w_ref)
        w = WeightedTaskSplitter.weight(alphas, X_p[i], y_p[i], X_s, y_s, eta=eta)
        ws.append(w)
    ws_ref = np.array(ws_ref).reshape(M, N)
    ws = np.array(ws)[:, 0].reshape(M, N)
    # np.testing.assert_equal(ws_ref, ws)
    np.testing.assert_almost_equal(ws_ref, ws, decimal=12)