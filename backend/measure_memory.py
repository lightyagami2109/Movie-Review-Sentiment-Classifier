"""Measure the highest memory the server uses while predicting.

    cd backend
    python measure_memory.py

It loads both models, then runs the heaviest requests the API accepts: a full batch of 200
long reviews on each model and word-by-word explanations. Run it after convert_checkpoint.py.
"""
import resource
import sys
from pathlib import Path

from fastapi.testclient import TestClient

import main

REVIEW = (
    "The film starts slowly but the performances are wonderful, the script is sharp and funny, "
    "and the final act is genuinely moving even if the middle drags and a few jokes fall flat. " * 3
)


def peak_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024  # macOS counts bytes, Linux KB


def check(response):
    if response.status_code != 200:
        sys.exit(f"Request failed ({response.status_code}): {response.text[:200]}")


def main_script():
    here = Path(__file__).resolve().parent
    for name in ("sst5_lstm_v1.npz", "sst5_tfidf_logreg_v2.joblib"):
        if not (here / name).is_file():
            sys.exit(f"{name} is missing. Run python convert_checkpoint.py first." if name.endswith("npz") else f"{name} is missing.")
    main.app.dependency_overrides[main.require_user] = lambda: {"sub": "measure"}
    print(f"Python started: {peak_mb():.0f} MB")
    with TestClient(main.app) as client:
        print(f"Both models loaded: {peak_mb():.0f} MB")
        for model in ("baseline", "lstm"):
            check(client.post("/api/v1/predict", json={"review": REVIEW, "model": model, "explain": True}))
            print(f"After one {model} review with explanations: {peak_mb():.0f} MB")
        for model in ("baseline", "lstm"):
            check(client.post("/api/v1/predict-batch", json={"reviews": [REVIEW] * 200, "model": model}))
            print(f"After a batch of 200 on {model}: {peak_mb():.0f} MB")
    print(f"\nPEAK MEMORY: {peak_mb():.0f} MB")


if __name__ == "__main__":
    main_script()
