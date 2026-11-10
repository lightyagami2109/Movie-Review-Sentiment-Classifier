"""Reproduce the fitted TF-IDF baseline from the SST-5 notebook."""
import argparse
from pathlib import Path
from subprocess import check_output

import joblib
from nltk import Tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, help="Local Stanford fiveclass/train.txt")
    args = parser.parse_args()
    url = "https://raw.githubusercontent.com/stanfordnlp/sentiment-treebank/master/fiveclass/train.txt"
    label_map = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
    data = args.train_file.read_bytes() if args.train_file else check_output(["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "60", url])
    trees = [Tree.fromstring(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if len(trees) != 8527:
        raise ValueError(f"Expected 8527 training excerpts, found {len(trees)}")
    texts = [" ".join(tree.leaves()) for tree in trees]
    labels = [label_map[int(tree.label())] for tree in trees]
    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    pipeline.fit(texts, labels)
    output = Path(__file__).with_name("sst5_tfidf_logreg_v1.joblib")
    joblib.dump(pipeline, output)
    print(output)


if __name__ == "__main__":
    main()
