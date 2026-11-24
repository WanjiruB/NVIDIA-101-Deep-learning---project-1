"""A tiny perceptron in pure NumPy.

Fill in the three TODO sections below. Do NOT rename these functions or
change their signatures — the tests call them by name.
"""
import numpy as np


def forward(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Predict labels for every row of X.

    Args:
        X: shape (n, d) — one input per row.
        w: shape (d,)   — weight vector.
        b: scalar bias.

    Returns:
        An int array of shape (n,) where each entry is +1 if w · x + b >= 0
        and -1 otherwise.
    """
    # TODO: compute z = X @ w + b, then return np.where(z >= 0, 1, -1)
    z = X @ w + b
    return np.where(z >= 0, 1, -1)


def update(w: np.ndarray, b: float, x_i: np.ndarray, y_i: int, y_hat: int, lr: float):
    """Apply the perceptron update rule for a single example.

    - If y_hat == y_i, return (w, b) unchanged.
    - Otherwise, return (w + lr * y_i * x_i, b + lr * y_i).
    """
    # TODO: implement the perceptron update rule

    if y_hat == y_i:
        return w, b

    w = w + lr * y_i * x_i
    b = b + lr * y_i

    return w, b


def train(X: np.ndarray, y: np.ndarray, epochs: int = 50, lr: float = 0.1, seed: int = 0):
    """Train the perceptron.

    Steps:
      1. Initialise w (small random values) and b = 0.
      2. For each epoch:
           - shuffle the row indices.
           - for each index i, compute y_hat_i from the CURRENT w, b.
           - call update() to refresh w, b.
         Then compute the accuracy of forward(X, w, b) against y and append
         it to `history` (as a Python float).
      3. Return (w, b, history).

    Use `np.random.default_rng(seed)` for reproducibility.
    """
    # TODO: implement the training loop
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=X.shape[1])
    b = 0.0
    history = []
    for epoch in range(epochs):
        indices = rng.permutation(len(X))
        for i in indices:
            x_i = X[i]
            y_i = y[i]

            y_hat = forward(X[i:i+1], w, b)[0]
            w, b = update(w, b, x_i, y_i, y_hat, lr)

        predictions = forward(X, w, b)
        accuracy = np.mean(predictions == y)
        history.append(float(accuracy))

    return w, b, history
