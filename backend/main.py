import json
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from torch import nn


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


@asynccontextmanager
async def lifespan(app):
    path = Path(os.getenv("SENTIMENT_MODEL_PATH", "sst5_lstm_v1.pt"))
    app.state.bundle = None
    if path.is_file():
        from nltk.tokenize import TreebankWordTokenizer

        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if bundle["tokenizer"] != "lowercase_treebank_v1":
            raise ValueError("Unsupported checkpoint tokenizer")
        config = bundle["model_config"]
        if config["num_classes"] != len(bundle["labels"]):
            raise ValueError("Checkpoint labels do not match the model output")
        model = SentimentLSTM(len(bundle["vocab"]), **config)
        model.load_state_dict(bundle["weights"])
        app.state.model = model.eval()
        app.state.tokenizer = TreebankWordTokenizer()
        app.state.bundle = bundle
    baseline_path = Path(os.getenv("SENTIMENT_BASELINE_PATH", "sst5_tfidf_logreg_v2.joblib"))
    if "SENTIMENT_BASELINE_PATH" not in os.environ and not baseline_path.is_file():
        baseline_path = baseline_path.with_name("sst5_tfidf_logreg_v1.joblib")
    app.state.baseline = None
    app.state.baseline_version = baseline_path.stem
    app.state.baseline_metrics = None
    if baseline_path.is_file():
        import joblib

        try:
            app.state.baseline = joblib.load(baseline_path)
        except Exception as error:  # e.g. file saved with a different scikit-learn version
            print(f"Could not load {baseline_path.name}: {error}")
            v1_path = baseline_path.with_name("sst5_tfidf_logreg_v1.joblib")
            if v1_path != baseline_path and v1_path.is_file():
                app.state.baseline = joblib.load(v1_path)
                app.state.baseline_version = v1_path.stem
    app.state.examples = None
    examples_path = Path(os.getenv("SENTIMENT_EXAMPLES_PATH", Path(__file__).with_name("examples.json")))
    if examples_path.is_file():
        app.state.examples = json.loads(examples_path.read_text(encoding="utf-8"))
    metrics_path = baseline_path.with_suffix(".metrics.json")
    if metrics_path.is_file():
        app.state.baseline_metrics = json.loads(metrics_path.read_text())
    yield


app = FastAPI(title="Movie Review Sentiment", lifespan=lifespan)


MAX_BATCH = 200


class BatchReviews(BaseModel):
    reviews: list[Annotated[str, Field(max_length=10_000)]] = Field(min_length=1, max_length=MAX_BATCH)
    model: Literal["lstm", "baseline"] = "baseline"


class Review(BaseModel):
    review: str = Field(min_length=1, max_length=10_000)
    model: Literal["lstm", "baseline"] = "lstm"
    explain: bool = False


def metadata(bundle):
    return {
        "model_version": bundle["model_version"],
        "labels": bundle["labels"],
        "dataset": bundle["dataset"],
        "test_metrics": bundle["test_metrics"],
    }


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "model_loaded": app.state.bundle is not None,
        "models": {
            "lstm": app.state.bundle is not None,
            "baseline": app.state.baseline is not None,
        },
    }


@app.get("/api/v1/model")
def model_info():
    bundle = app.state.bundle
    if bundle is None:
        raise HTTPException(503, "Model checkpoint is unavailable")
    info = metadata(bundle)
    info["baseline_metrics"] = getattr(app.state, "baseline_metrics", None)
    return info


BASELINE_LABELS = ["negative", "neutral", "positive"]
MAX_EXPLAINED_WORDS = 80


def class_probabilities(model, texts):
    """Return one {label: probability} dict per text for the chosen model."""
    if model == "baseline":
        baseline = app.state.baseline
        scores = baseline.predict_proba(texts)
        names = [BASELINE_LABELS[int(label)] for label in baseline.classes_]
        return [dict(zip(names, map(float, row))) for row in scores]
    bundle = app.state.bundle
    length = bundle["max_length"]
    batch = []
    for text in texts:
        tokens = app.state.tokenizer.tokenize(
            text.lower().replace("(", " -lrb- ").replace(")", " -rrb- ")
        )
        ids = [bundle["vocab"].get(token, 1) for token in tokens[:length]] or [1]
        batch.append(ids + [0] * (length - len(ids)))
    scores = app.state.model(torch.tensor(batch)).softmax(dim=1)
    return [dict(zip(bundle["labels"], row)) for row in scores.tolist()]


def lean(probabilities):
    """Log-odds of positive vs negative. Unlike raw probabilities it does not
    flatten at 99%, so single words still show their effect."""
    return math.log(probabilities["positive"] + 1e-6) - math.log(probabilities["negative"] + 1e-6)


def explain_words(model, text, full):
    """Leave-one-out: how far does dropping each word move the lean? Positive
    impact means the word pushes the review toward positive."""
    words = text.split()
    shown = words[:MAX_EXPLAINED_WORDS]
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(shown))]
    base = lean(full)
    impacts = [0.0] * len(shown)
    if len(words) > 1:
        for i, probabilities in enumerate(class_probabilities(model, variants)):
            impacts[i] = base - lean(probabilities)
    return [{"text": word, "impact": round(impact, 3)} for word, impact in zip(shown, impacts)] + [
        {"text": word, "impact": 0.0} for word in words[MAX_EXPLAINED_WORDS:]
    ]


@app.get("/api/v1/examples")
def examples():
    """Real test-set sentences for the website: a random sample and the model's confident mistakes."""
    if getattr(app.state, "examples", None) is None:
        raise HTTPException(404, "Example reviews are unavailable. Run python build_examples.py in the backend folder.")
    return app.state.examples


def require_model(model):
    if model == "baseline":
        if app.state.baseline is None:
            raise HTTPException(503, "TF-IDF baseline checkpoint is unavailable")
    elif app.state.bundle is None:
        raise HTTPException(503, "Model checkpoint is unavailable")


@app.post("/api/v1/predict-batch")
@torch.inference_mode()
def predict_batch(payload: BatchReviews):
    texts = [review.strip() for review in payload.reviews]
    if any(not text for text in texts):
        raise HTTPException(422, "Every review needs at least one non-whitespace character")
    require_model(payload.model)
    results = class_probabilities(payload.model, texts)
    return {"results": [
        {"review": text, "sentiment": max(probabilities, key=probabilities.get), "probabilities": probabilities}
        for text, probabilities in zip(texts, results)
    ]}


@app.post("/api/v1/predict")
@torch.inference_mode()
def predict(payload: Review):
    text = payload.review.strip()
    if not text:
        raise HTTPException(422, "Enter a review with at least one non-whitespace character")
    if payload.model == "baseline":
        if app.state.baseline is None:
            raise HTTPException(503, "TF-IDF baseline checkpoint is unavailable")
        version = getattr(app.state, "baseline_version", "sst5_tfidf_logreg_v2")
    else:
        if app.state.bundle is None:
            raise HTTPException(503, "Model checkpoint is unavailable")
        version = app.state.bundle["model_version"]
    probabilities = class_probabilities(payload.model, [text])[0]
    response = {
        "sentiment": max(probabilities, key=probabilities.get),
        "probabilities": probabilities,
        "model_version": version,
    }
    if payload.explain:
        response["words"] = explain_words(payload.model, text, probabilities)
    return response
