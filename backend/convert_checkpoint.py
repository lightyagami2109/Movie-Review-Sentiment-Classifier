"""Turn the PyTorch checkpoint into a NumPy file the deployed server can read.

Run once on your own computer (it needs PyTorch, the server does not):

    cd backend
    python convert_checkpoint.py

It writes sst5_lstm_v1.npz next to the .pt file, then checks the NumPy version against
PyTorch on random inputs and refuses to finish if the answers differ.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from numpy_lstm import WEIGHT_NAMES, NumpyLSTM


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, ids):
        states, _ = self.lstm(self.embedding(ids))
        last = (ids != 0).sum(dim=1) - 1
        return self.output(states[torch.arange(ids.size(0)), last])


def main(source, target):
    bundle = torch.load(source, map_location="cpu", weights_only=True)
    state = bundle["weights"]
    arrays = {
        "embedding": state["embedding.weight"],
        "w_ih": state["lstm.weight_ih_l0"],
        "w_hh": state["lstm.weight_hh_l0"],
        "b_ih": state["lstm.bias_ih_l0"],
        "b_hh": state["lstm.bias_hh_l0"],
        "w_out": state["output.weight"],
        "b_out": state["output.bias"],
    }
    arrays = {name: value.detach().cpu().numpy().astype(np.float32) for name, value in arrays.items()}
    meta = {key: value for key, value in bundle.items() if key != "weights"}
    meta_text = json.dumps(meta)  # fails loudly if something in the checkpoint is not plain data
    np.savez_compressed(target, meta=np.array(meta_text), **arrays)

    # Check: same answers as PyTorch on random reviews of random lengths.
    model = SentimentLSTM(len(bundle["vocab"]), **bundle["model_config"])
    model.load_state_dict(state)
    model.eval()
    reloaded = {name: np.load(target)[name] for name in WEIGHT_NAMES}
    fast = NumpyLSTM(reloaded)
    rng = np.random.default_rng(0)
    steps = bundle["max_length"]
    ids = np.zeros((64, steps), dtype=np.int64)
    for row in ids:
        length = int(rng.integers(1, steps + 1))
        row[:length] = rng.integers(1, len(bundle["vocab"]), size=length)
    with torch.inference_mode():
        expected = model(torch.tensor(ids)).softmax(dim=1).numpy()
    actual = fast.probabilities(ids)
    worst = float(np.abs(expected - actual).max())
    print(f"Largest difference from PyTorch: {worst:.2e}")
    if worst > 1e-4:
        target.unlink()
        sys.exit("The NumPy version does not match PyTorch, so nothing was kept. Send this message to Claude.")
    print(f"OK. Wrote {target.name} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    main(folder / "sst5_lstm_v1.pt", folder / "sst5_lstm_v1.npz")
