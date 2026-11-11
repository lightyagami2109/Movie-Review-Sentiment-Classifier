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
- POST /api/v1/predict accepts a JSON body such as `{"review":"The acting was wonderful."}`.

Place your Colab export `sst5_lstm_v1.pt` in this folder, or set
`SENTIMENT_MODEL_PATH` to its location before starting the server. Without
the checkpoint, health still works and the prediction and model routes
return HTTP 503. Restart the server to load a replacement checkpoint.
A replacement dataset can use this API if its exported bundle follows
the same model and tokenizer format.

## Enable TF-IDF predictions

From this folder, run `python train_baseline.py` after installing the
requirements. It fits the notebook's baseline on the SST-5 training split
and saves the vectorizer and classifier together. Restart FastAPI to load
the generated file, or configure its path with `SENTIMENT_BASELINE_PATH`.
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
