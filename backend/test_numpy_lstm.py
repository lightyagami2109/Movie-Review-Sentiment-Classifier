"""Run with python -m unittest -v test_numpy_lstm from backend/."""
import unittest

import numpy as np

import numpy_lstm


def random_weights(vocab=30, emb=6, hidden=5, classes=3, seed=1):
    rng = np.random.default_rng(seed)
    make = lambda *shape: rng.normal(0, 0.5, shape).astype(np.float32)
    weights = {"embedding": make(vocab, emb), "w_ih": make(4 * hidden, emb), "w_hh": make(4 * hidden, hidden),
               "b_ih": make(4 * hidden), "b_hh": make(4 * hidden), "w_out": make(classes, hidden), "b_out": make(classes)}
    weights["embedding"][0] = 0
    return weights


def reference(weights, ids):
    """One review at a time, written as plainly as possible from the LSTM equations."""
    sig = lambda x: 1 / (1 + np.exp(-x))
    hidden = weights["w_hh"].shape[1]
    out = []
    for row in ids:
        length = int((row != 0).sum())
        h = np.zeros(hidden)
        c = np.zeros(hidden)
        for token in row[:length]:
            x = weights["embedding"][token].astype(np.float64)
            z = weights["w_ih"] @ x + weights["b_ih"] + weights["w_hh"] @ h + weights["b_hh"]
            i, f, g, o = (z[k * hidden:(k + 1) * hidden] for k in range(4))
            c = sig(f) * c + sig(i) * np.tanh(g)
            h = sig(o) * np.tanh(c)
        out.append(weights["w_out"] @ h + weights["b_out"])
    return np.array(out)


class NumpyLstmTests(unittest.TestCase):
    def test_matches_a_plain_reference_on_padded_batches(self):
        weights = random_weights()
        rng = np.random.default_rng(2)
        ids = np.zeros((12, 9), dtype=np.int64)
        for row in ids:
            length = int(rng.integers(1, 10))
            row[:length] = rng.integers(1, 30, size=length)
        model = numpy_lstm.NumpyLSTM(weights)
        np.testing.assert_allclose(model.logits(ids), reference(weights, ids), atol=1e-4)

    def test_padding_does_not_change_the_answer(self):
        model = numpy_lstm.NumpyLSTM(random_weights())
        short = np.array([[4, 7, 2, 0, 0]])
        long = np.array([[4, 7, 2, 0, 0, 0, 0, 0, 0]])
        np.testing.assert_allclose(model.logits(short), model.logits(long), atol=1e-6)

    def test_a_review_is_independent_of_the_others_in_its_batch(self):
        model = numpy_lstm.NumpyLSTM(random_weights())
        a, b = np.array([[3, 9, 1, 0]]), np.array([[5, 5, 5, 5]])
        together = model.logits(np.vstack([a, b]))
        np.testing.assert_allclose(together[0], model.logits(a)[0], atol=1e-6)
        np.testing.assert_allclose(together[1], model.logits(b)[0], atol=1e-6)

    def test_probabilities_sum_to_one(self):
        probabilities = numpy_lstm.NumpyLSTM(random_weights()).probabilities(np.array([[1, 2, 3], [4, 0, 0]]))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue((probabilities >= 0).all())

    def test_loading_rejects_unknown_tokenizers(self):
        import json
        import tempfile
        from pathlib import Path

        meta = {"tokenizer": "something_else", "model_config": {"num_classes": 3}, "labels": ["a", "b", "c"]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.npz"
            np.savez(path, meta=np.array(json.dumps(meta)), **random_weights())
            with self.assertRaises(ValueError):
                numpy_lstm.load(path)

    def test_loading_round_trip(self):
        import json
        import tempfile
        from pathlib import Path

        meta = {"tokenizer": "lowercase_treebank_v1", "model_config": {"num_classes": 3}, "labels": ["a", "b", "c"], "vocab": {"x": 2}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.npz"
            np.savez(path, meta=np.array(json.dumps(meta)), **random_weights())
            model, bundle = numpy_lstm.load(path)
        self.assertEqual(bundle["vocab"], {"x": 2})
        self.assertEqual(model.logits(np.array([[1, 2]])).shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
