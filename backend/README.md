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
