import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

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
    baseline_path = Path(os.getenv("SENTIMENT_BASELINE_PATH", "sst5_tfidf_logreg_v1.joblib"))
    app.state.baseline = None
    if baseline_path.is_file():
        import joblib

        app.state.baseline = joblib.load(baseline_path)
    yield


app = FastAPI(title="Movie Review Sentiment", lifespan=lifespan)


class Review(BaseModel):
    review: str = Field(min_length=1, max_length=10_000)
    model: Literal["lstm", "baseline"] = "lstm"


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
    return metadata(bundle)


@app.post("/api/v1/predict")
@torch.inference_mode()
def predict(payload: Review):
    text = payload.review.strip()
    if not text:
        raise HTTPException(422, "Enter a review with at least one non-whitespace character")
    if payload.model == "baseline":
        baseline = app.state.baseline
        if baseline is None:
            raise HTTPException(503, "TF-IDF baseline checkpoint is unavailable")
        labels = ["negative", "neutral", "positive"]
        scores = baseline.predict_proba([text])[0]
        return {
            "sentiment": labels[int(baseline.classes_[scores.argmax()])],
            "probabilities": {labels[int(label)]: float(score) for label, score in zip(baseline.classes_, scores)},
            "model_version": "sst5_tfidf_logreg_v1",
        }
    bundle = app.state.bundle
    if bundle is None:
        raise HTTPException(503, "Model checkpoint is unavailable")

    tokens = app.state.tokenizer.tokenize(
        text.lower().replace("(", " -lrb- ").replace(")", " -rrb- ")
    )
    length = bundle["max_length"]
    ids = [bundle["vocab"].get(token, 1) for token in tokens[:length]] or [1]
    ids += [0] * (length - len(ids))
    scores = app.state.model(torch.tensor([ids])).softmax(dim=1)[0]
    labels = bundle["labels"]
    return {
        "sentiment": labels[scores.argmax().item()],
        "probabilities": dict(zip(labels, scores.tolist())),
        "model_version": bundle["model_version"],
    }
