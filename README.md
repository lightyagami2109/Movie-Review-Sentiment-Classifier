# Movie Review Sentiment Classifier

A PyTorch LSTM classifies movie-review excerpts as negative, neutral, or positive. FastAPI serves the trained checkpoint locally.

## Training and evaluation

Open [the SST-5 notebook](notebooks/sst5_three_class.ipynb) in Google Colab and run the sections in order. It downloads the Stanford Sentiment Treebank, groups the original five labels into three, and compares the LSTM with a TF-IDF + logistic regression baseline. The notebook selects the LSTM checkpoint by validation macro F1.

| Model | Test accuracy | Test macro F1 |
| --- | ---: | ---: |
| TF-IDF + logistic regression | 61.6% | 0.550 |
| PyTorch LSTM | 53.3% | 0.488 |

These scores are from the same held-out SST-5 test split. The API serves the LSTM.

## Run the API

Download `sst5_lstm_v1.pt` after running the notebook's checkpoint export. Follow the [backend setup instructions](backend/README.md) to install dependencies, load the checkpoint, and try a prediction.

The [original Naive Bayes notebook](MRSCNaiveBayes.ipynb) is a separate binary sentiment experiment.

## Run the React frontend

With FastAPI running on port 8000, open a second terminal and run:

```sh
cd frontend
npm ci
npm run dev
```

Open the address Vite prints and submit a review. The development server
forwards API requests to FastAPI. The form displays the active model version
and the labels and scores returned by the API.
