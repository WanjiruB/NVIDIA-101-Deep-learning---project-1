# Perceptron From Scratch

A single-neuron classifier implemented with nothing but NumPy. This is the smallest possible neural network — and the same update pattern is inside every neuron of every modern deep model.

## Quickstart

```bash
bash run.sh  # trains the perceptron and prints final accuracy
bash test.sh # runs the graded pytest suite
```

## Your task

Open `perceptron.py` and fill in the three functions marked `TODO`:

1. `forward(X, w, b)` — predictions (+1 / -1) for every row of X
2. `update(w, b, x_i, y_i, y_hat, lr)` — the perceptron update rule
3. `train(X, y, epochs, lr)` — loop over epochs, apply updates, record accuracy

`main.py` and `tests/` are already written — do not modify them.

## Reflection
The perceptron update rule adjusts the weights and bias whenever the model makes a wrong prediction. Geometrically, this moves the decision boundary to make it more likely to classify that example correctly.

A single perceptron cannot learn the XOR dataset because its points cannot be separated correctly using one straight line. No matter how many times we train it, one straight decision boundary cannot solve that problem.
