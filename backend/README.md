# Movie Review Sentiment API

From this folder, install dependencies and start the API:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs to try the endpoints.

- GET /api/v1/health shows whether a checkpoint loaded.
- GET /api/v1/model provides the active labels, dataset and evaluation metrics.
- POST /api/v1/predict accepts a JSON body such as `{"review":"The acting was wonderful.","model":"baseline","explain":true}`. `model` is `lstm` or `baseline`; `explain` adds the per-word effect (leave-one-out, in log-odds).
- POST /api/v1/predict-batch accepts `{"reviews":["...","..."],"model":"baseline"}` (1 to 200 reviews) and returns one prediction per review.
- GET /api/v1/examples returns real test-set reviews for the website (`examples.json`, built by `python build_examples.py`).

Place your Colab export `sst5_lstm_v1.pt` in this folder, or set
`SENTIMENT_MODEL_PATH` to its location before starting the server. Without
the checkpoint, health still works and the prediction and model routes
return HTTP 503. Restart the server to load a replacement checkpoint.
A replacement dataset can use this API if its exported bundle follows
the same model and tokenizer format.

## Enable TF-IDF predictions

From this folder, run `python train_baseline.py` after installing the
requirements. It fits the baseline on every labelled phrase in the SST-5
training trees (about 150k examples, so single words like "fabulous" are
learned), prints validation and test scores on whole sentences, and saves
`sst5_tfidf_logreg_v2.joblib` (vectorizer and classifier together). Use
`--sentences-only` for the original v1 baseline. Restart FastAPI to load
the generated file, or configure its path with `SENTIMENT_BASELINE_PATH`.
If v2 is missing, the server falls back to a v1 file.
Load only a baseline artifact you created or trust.

The prediction request accepts an optional `model` field with value
`lstm` (default) or `baseline`.

## Run API tests

From this folder, run these commands in your Python environment.

```sh
python -m pip install -r requirements-dev.txt
python -m unittest -v test_api
```

These tests check independent model availability, missing checkpoints,
the default model choice, and invalid requests. They run without model files.
Model accuracy and real checkpoint inference are evaluated separately.
