"""The trained LSTM, run with NumPy only (no PyTorch needed to serve predictions).

Same maths as PyTorch's nn.Embedding -> nn.LSTM(batch_first) -> nn.Linear, reading the
state at each review's last real word. Gate order follows PyTorch: input, forget, cell, output.
"""
import json

import numpy as np

WEIGHT_NAMES = ("embedding", "w_ih", "w_hh", "b_ih", "b_hh", "w_out", "b_out")


def sigmoid(values):
    return 0.5 * (1.0 + np.tanh(0.5 * values))  # stable for large positive and negative inputs


class NumpyLSTM:
    def __init__(self, weights):
        self.embedding = weights["embedding"]  # (vocab, embedding_dim)
        self.w_ih = weights["w_ih"]  # (4 * hidden, embedding_dim)
        self.w_hh = weights["w_hh"]  # (4 * hidden, hidden)
        self.bias = weights["b_ih"] + weights["b_hh"]
        self.w_out = weights["w_out"]  # (classes, hidden)
        self.b_out = weights["b_out"]
        self.hidden = self.w_hh.shape[1]
        self.w_ih_t = np.ascontiguousarray(self.w_ih.T)
        self.w_hh_t = np.ascontiguousarray(self.w_hh.T)

    def logits(self, ids):
        """ids: integer array (batch, steps), padded with 0. Returns (batch, classes)."""
        ids = np.asarray(ids)
        batch, steps = ids.shape
        # One matrix product for every step's input part; only the recurrent part needs the loop.
        inputs = self.embedding[ids] @ self.w_ih_t + self.bias  # (batch, steps, 4 * hidden)
        h = np.zeros((batch, self.hidden), dtype=inputs.dtype)
        c = np.zeros_like(h)
        last = np.maximum((ids != 0).sum(axis=1) - 1, 0)
        picked = np.zeros_like(h)
        rows = np.arange(batch)
        for step in range(steps):
            gates = inputs[:, step] + h @ self.w_hh_t
            i, f, g, o = np.split(gates, 4, axis=1)
            c = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
            h = sigmoid(o) * np.tanh(c)
            here = last == step
            if here.any():
                picked[rows[here]] = h[here]
        return picked @ self.w_out.T + self.b_out

    def probabilities(self, ids):
        scores = self.logits(ids)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)


def load(path):
    """Read a .npz written by convert_checkpoint.py. Returns (model, bundle) where bundle
    holds the vocabulary, labels, metrics and other settings that were saved with the weights."""
    with np.load(path, allow_pickle=False) as data:
        bundle = json.loads(str(data["meta"]))
        weights = {name: data[name].astype(np.float32) for name in WEIGHT_NAMES}
    if bundle["tokenizer"] != "lowercase_treebank_v1":
        raise ValueError("Unsupported checkpoint tokenizer")
    if bundle["model_config"]["num_classes"] != len(bundle["labels"]):
        raise ValueError("Checkpoint labels do not match the model output")
    return NumpyLSTM(weights), bundle
