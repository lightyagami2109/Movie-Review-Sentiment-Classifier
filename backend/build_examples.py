"""Build examples.json for the website from the SST-5 test split.

Contains (1) a random sample of real test sentences with their dataset label and
(2) test sentences the TF-IDF model gets wrong with high confidence. Re-run this
after retraining so the "Where it fails" list matches the current model:

    python build_examples.py
"""
import argparse
import json
import random
import re
from pathlib import Path

import joblib

from train_baseline import LABEL_NAMES, load_trees, sentence_examples

SAMPLE_SIZE = 300


def detokenize(text):
    """Turn Treebank tokens ("It 's too long , ...") back into normal text."""
    text = text.replace("-LRB-", "(").replace("-RRB-", ")").replace("``", '"').replace("''", '"')
    text = re.sub(r"\s+([.,;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[$])\s+", r"\1", text)
    text = re.sub(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", text)
    text = re.sub(r"(\w) - (\w)", r"\1-\2", text)
    text = re.sub(r'"\s+([^"]*?)\s+"', r'"\1"', text)
    return text.strip()
FAILURE_COUNT = 60


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, help="Local Stanford fiveclass/test.txt")
    args = parser.parse_args()

    folder = Path(__file__).parent
    model_path = folder / "sst5_tfidf_logreg_v2.joblib"
    pipeline = joblib.load(model_path)
    examples = [(detokenize(text), label) for text, label in sentence_examples(load_trees("test", args.test_file))]
    texts = [text for text, _ in examples]
    probabilities = pipeline.predict_proba(texts)

    rng = random.Random(7)
    sample = rng.sample(examples, SAMPLE_SIZE)
    failures = []
    for (text, label), row in zip(examples, probabilities):
        predicted = int(row.argmax())
        if predicted != label and len(text.split()) >= 6:
            failures.append({"text": text, "label": LABEL_NAMES[label],
                             "predicted": LABEL_NAMES[predicted], "confidence": round(float(row.max()), 3)})
    failures.sort(key=lambda item: -item["confidence"])

    output = {
        "source": "Stanford Sentiment Treebank, test split (never used for training)",
        "samples": [{"text": text, "label": LABEL_NAMES[label]} for text, label in sample],
        "failures": failures[:FAILURE_COUNT],
    }
    (folder / "examples.json").write_text(json.dumps(output, indent=1, ensure_ascii=False))
    print(f"Saved {len(output['samples'])} samples and {len(output['failures'])} failures to examples.json")


if __name__ == "__main__":
    main()
