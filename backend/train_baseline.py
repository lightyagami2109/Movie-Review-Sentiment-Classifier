"""Train the TF-IDF + logistic regression baseline on SST-5 (3 classes).

By default the model learns from every labelled phrase in the Stanford
Sentiment Treebank parse trees (about 150k examples), not just the 8,527
whole sentences. Single words such as "fabulous" are labelled in the
treebank, so the model can learn that they are positive. Pass
--sentences-only to reproduce the original v1 baseline.
"""
import argparse
import json
from pathlib import Path
from subprocess import check_output

import joblib
from nltk import Tree
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline

BASE_URL = "https://raw.githubusercontent.com/stanfordnlp/sentiment-treebank/master/fiveclass/"
LABEL_MAP = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
LABEL_NAMES = ["negative", "neutral", "positive"]
EXPECTED_SENTENCES = {"train": 8527, "dev": 1095, "test": 2203}


def load_trees(split, local_file=None):
    if local_file:
        data = local_file.read_bytes()
    else:
        data = check_output(["curl", "--fail", "--silent", "--show-error", "--location",
                             "--max-time", "120", f"{BASE_URL}{split}.txt"])
    trees = [Tree.fromstring(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if len(trees) != EXPECTED_SENTENCES[split]:
        raise ValueError(f"Expected {EXPECTED_SENTENCES[split]} {split} excerpts, found {len(trees)}")
    return trees


def sentence_examples(trees):
    """One (text, label) pair per sentence: the original v1 training data."""
    return [(" ".join(t.leaves()), LABEL_MAP[int(t.label())]) for t in trees]


def phrase_examples(trees):
    """One (text, label) pair per labelled phrase, including whole sentences."""
    seen = set()
    examples = []
    for tree in trees:
        for node in tree.subtrees():
            text = " ".join(node.leaves())
            label = LABEL_MAP[int(node.label())]
            if (text, label) not in seen:  # drop exact duplicates ("the", "a", ...)
                seen.add((text, label))
                examples.append((text, label))
    return examples


# Filler words ("the", "was", "it") are labelled neutral thousands of times in
# the phrase data and drown out short reviews like "the movie was fabulous".
# Words that change meaning (not, but, very, only...) are kept.
KEEP_WORDS = {"not", "no", "nor", "never", "cannot", "nothing", "without", "but", "very", "too",
              "most", "more", "less", "least", "few", "many", "much", "only", "however",
              "although", "though", "against", "hardly", "neither", "none", "nobody",
              "nowhere", "still", "even"}
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) - KEEP_WORDS)


def build_pipeline():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=STOP_WORDS),
        LogisticRegression(C=3, max_iter=2000, class_weight="balanced"),
    )


def evaluate(pipeline, name, trees):
    texts, labels = zip(*sentence_examples(trees))
    predicted = pipeline.predict(list(texts))
    accuracy = accuracy_score(labels, predicted)
    macro_f1 = f1_score(labels, predicted, average="macro")
    print(f"\n{name}: accuracy {accuracy:.3f}, macro F1 {macro_f1:.3f}")
    print(classification_report(labels, predicted, target_names=LABEL_NAMES, digits=2))
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, help="Local Stanford fiveclass/train.txt")
    parser.add_argument("--dev-file", type=Path, help="Local Stanford fiveclass/dev.txt")
    parser.add_argument("--test-file", type=Path, help="Local Stanford fiveclass/test.txt")
    parser.add_argument("--sentences-only", action="store_true",
                        help="Train on whole sentences only (the original v1 baseline)")
    args = parser.parse_args()

    train_trees = load_trees("train", args.train_file)
    examples = sentence_examples(train_trees) if args.sentences_only else phrase_examples(train_trees)
    texts, labels = zip(*examples)
    print(f"Training on {len(texts)} {'sentences' if args.sentences_only else 'labelled phrases'}")

    pipeline = build_pipeline()
    pipeline.fit(list(texts), list(labels))

    evaluate(pipeline, "Validation (whole sentences)", load_trees("dev", args.dev_file))
    test_metrics = evaluate(pipeline, "Test (whole sentences)", load_trees("test", args.test_file))

    name = "sst5_tfidf_logreg_v1.joblib" if args.sentences_only else "sst5_tfidf_logreg_v2.joblib"
    output = Path(__file__).with_name(name)
    joblib.dump(pipeline, output)
    # The API shows these scores on the results page next to the LSTM's.
    output.with_suffix(".metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
