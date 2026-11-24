import numpy as np
from perceptron import forward, update, train


def _linearly_separable(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X_pos = rng.normal(loc=(2, 2), scale=0.7, size=(n // 2, 2))
    X_neg = rng.normal(loc=(-2, -2), scale=0.7, size=(n // 2, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n // 2) + [-1] * (n // 2))
    idx = rng.permutation(n)
    return X[idx], y[idx]


def test_forward_shape_and_values():
    X = np.array([[1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]])
    w = np.array([1.0, 1.0])
    b = 0.0
    out = forward(X, w, b)
    assert out.shape == (3,), f"expected (3,), got {out.shape}"
    assert set(np.unique(out).tolist()).issubset({1, -1}), "predictions must be +1 or -1"
    assert out[0] == 1 and out[1] == -1


def test_update_no_op_when_correct():
    w = np.array([1.0, 1.0])
    b = 0.5
    w2, b2 = update(w, b, np.array([1.0, 1.0]), 1, 1, lr=0.1)
    assert np.allclose(w2, w) and b2 == b, "no update when prediction is correct"


def test_update_moves_boundary_when_wrong():
    w = np.array([0.0, 0.0])
    b = 0.0
    w2, b2 = update(w, b, np.array([1.0, 2.0]), 1, -1, lr=0.5)
    assert np.allclose(w2, [0.5, 1.0]), f"expected [0.5, 1.0], got {w2}"
    assert b2 == 0.5


def test_training_accuracy_high_on_separable_data():
    X, y = _linearly_separable(n=300, seed=1)
    w, b, hist = train(X, y, epochs=30, lr=0.1, seed=0)
    preds = forward(X, w, b)
    acc = float((preds == y).mean())
    assert acc >= 0.97, f"expected >=0.97 training accuracy, got {acc:.3f}"


def test_history_records_each_epoch():
    X, y = _linearly_separable(n=100, seed=2)
    _, _, hist = train(X, y, epochs=15, lr=0.1, seed=0)
    assert len(hist) == 15, f"expected 15 entries in history, got {len(hist)}"
    for h in hist:
        assert 0.0 <= h <= 1.0, f"accuracy out of range: {h}"


def test_training_improves_from_random_init():
    X, y = _linearly_separable(n=200, seed=3)
    rng = np.random.default_rng(999)
    w0 = rng.normal(size=2); b0 = 0.0
    pre_preds = forward(X, w0, b0)
    pre_acc = float((pre_preds == y).mean())

    w, b, _ = train(X, y, epochs=25, lr=0.1, seed=0)
    post_preds = forward(X, w, b)
    post_acc = float((post_preds == y).mean())
    assert post_acc > pre_acc + 0.1, (
        f"training did not clearly improve accuracy: {pre_acc:.3f} -> {post_acc:.3f}"
    )
