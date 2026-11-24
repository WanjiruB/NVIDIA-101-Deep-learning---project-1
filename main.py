"""Train the perceptron on a toy dataset and print the final accuracy.

Run me with:  bash run.sh
"""
import numpy as np
from perceptron import forward, train


def make_data(n: int = 200, seed: int = 0):
    """Two Gaussian blobs (the +1 and -1 classes), off-centre so the bias has to be learned too."""
    rng = np.random.default_rng(seed)
    X_pos = rng.normal(loc=(5, 5), scale=0.5, size=(n // 2, 2))
    X_neg = rng.normal(loc=(3, 3), scale=0.5, size=(n // 2, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n // 2) + [-1] * (n // 2))
    idx = rng.permutation(n)
    return X[idx], y[idx]


def main():
    X, y = make_data(n=200, seed=0)
    w0 = np.random.default_rng(1).normal(size=X.shape[1])
    pre_acc = float((forward(X, w0, 0.0) == y).mean())
    print(f"Accuracy before training (random weights): {pre_acc:.3f}")
    w, b, hist = train(X, y, epochs=30, lr=0.1, seed=0)
    preds = forward(X, w, b)
    acc = float((preds == y).mean())
    print(f"Final training accuracy: {acc:.3f}")
    print(f"Weights: {w}, bias: {b:.3f}")
    print(f"Accuracy history (first 5 epochs): {[round(h, 3) for h in hist[:5]]}")
    print(f"Accuracy history (last 5 epochs):  {[round(h, 3) for h in hist[-5:]]}")
    if acc >= 0.97:
        print("\n✓ Nice — the perceptron converged on this separable dataset.")
    else:
        print("\n✗ Accuracy is below 0.97 — check your update() and train() logic.")


if __name__ == "__main__":
    main()
