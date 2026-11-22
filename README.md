# Movie Review Sentiment Classifier

Classify movie-review excerpts as negative, neutral, or positive using a PyTorch LSTM or TF-IDF + Logistic Regression, plus a fine-tuned DistilBERT experiment with an inference speed benchmark. A React website (Review Reader) sends reviews to a FastAPI backend.

## Quick start

From this folder, after the model files are in `backend/` (see below):

```sh
bash start.sh
```

It finds the Python environment, checks the packages, starts the API on port 8000 and the website on port 5173, and opens the site. Press Ctrl+C to stop both.

The website can:
- classify one review and show which words moved the score,
- try random real reviews from the test set and compare the answer with the dataset label,
- list reviews the TF-IDF model gets wrong ("Where it fails"),
- keep your last 5 reviews, and
- analyze up to 200 reviews at once from pasted text or a CSV file and download the predictions.

## Training and evaluation

Open [the SST-5 notebook](notebooks/sst5_three_class.ipynb) in Google Colab and run the sections in order. It downloads the Stanford Sentiment Treebank, groups the original five labels into three, and compares the LSTM with a TF-IDF + logistic regression baseline. The notebook selects the LSTM checkpoint by validation macro F1.

| Model | Test accuracy | Test macro F1 |
| --- | ---: | ---: |
| TF-IDF + Logistic Regression | 64.5% | 0.579 |
| PyTorch LSTM | 53.3% | 0.488 |

These scores are from the same held-out SST-5 test split of 2,203 reviews, which no model trained on. The website uses TF-IDF + Logistic Regression by default, and you can switch to the LSTM in the review form.

The TF-IDF model used by the website is trained by `backend/train_baseline.py` on every labelled phrase of the training sentences (about 159,000 examples, from the same Stanford training file). The notebook's own baseline section trains on whole sentences only and scores lower (61.6% accuracy, 0.550 macro F1).

## DistilBERT: fine-tuning and faster inference

[The DistilBERT notebook](notebooks/distilbert_sst3_benchmark.ipynb) fine-tunes `distilbert-base-uncased` on the same files, label grouping, class-weighted loss and checkpoint rule (best validation macro F1) as the LSTM, then benchmarks inference. Run it in Colab with a T4 GPU (Runtime → Run all, about 5 minutes).

Results from one run on a Colab Tesla T4 (torch 2.14.1, transformers 5.22.1, seed 42):

| Model (same 2,203-sentence test split) | Test accuracy | Test macro F1 |
| --- | ---: | ---: |
| PyTorch LSTM | 53.3% | 0.488 |
| TF-IDF + Logistic Regression (re-scored in the same run) | 64.5% | 0.579 |
| **DistilBERT** | **71.2%** | **0.660** |

Training used fp16 mixed precision (`torch.autocast` + `GradScaler`), AdamW at 2e-5 with linear warm-up, batch size 32 and 4 epochs: 69 seconds in total, 1.46 GB peak GPU memory. Epoch 4 had the best validation macro F1 (0.665). Neutral is still the weakest class (F1 0.382).

Inference over all 2,203 test sentences, tokenization included (fastest of 3 runs each):

| Setup | Sentences/s | ms per sentence | GPU memory (weights + working) | Macro F1 | Same labels as fp32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before: fp32, batch 1, padded to 128 | 172 | 5.82 | 255 + 5 MB | 0.660 | 100% |
| fp32, batch 64, dynamic padding | 655 | 1.53 | 255 + 129 MB | 0.660 | 100% |
| fp16 autocast, batch 64, dynamic padding | 2,170 | 0.46 | 255 + 82 MB | 0.660 | 100% |
| **After: fp16 weights, batch 64, length-sorted** | **4,618** | **0.22** | **128 + 56 MB** | **0.660** | **100%** |

Batching, padding each batch only to its longest sentence, sorting sentences by length and running in fp16 made inference **26.9× faster** and halved the weight memory, with every predicted label identical to the fp32 baseline. Speed depends on the GPU and library versions, so rerun the notebook to reproduce the numbers on your hardware.

## Prepare both models

Run the Colab notebook and download `sst5_lstm_v1.pt` into `backend/`.
After installing the backend dependencies, run `python train_baseline.py`
from `backend/` (about 20 seconds). It trains on phrase-level labels, prints validation and test scores, and creates `sst5_tfidf_logreg_v2.joblib` and its scores file. Then run `python build_examples.py` to refresh the example and "Where it fails" reviews for the website.

Alternatively, configure the file locations using `SENTIMENT_MODEL_PATH`
and `SENTIMENT_BASELINE_PATH`. Restart FastAPI after adding either model.
The LSTM export includes its vocabulary and settings. The baseline file
contains the fitted vectorizer and classifier together.

## Limitations

SST-5 contains short review excerpts. Neutral sentiment is the weakest class
for both models in this run. Individual prediction scores are model estimates.

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
forwards API requests to FastAPI. `bash start.sh` does all of this in one step.
