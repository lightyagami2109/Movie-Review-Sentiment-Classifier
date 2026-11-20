"""Run with python -m unittest -v test_api from backend/."""
import unittest
from unittest.mock import patch

import numpy

from fastapi.testclient import TestClient

from main import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        # Tests run without downloading or loading trained model artifacts.
        with patch("main.Path.is_file", return_value=False):
            self.client = TestClient(app)
            self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_health_reports_each_model_independently(self):
        for lstm, baseline in [(True, True), (True, False), (False, True), (False, False)]:
            with self.subTest(lstm=lstm, baseline=baseline):
                app.state.bundle = {} if lstm else None
                app.state.baseline = object() if baseline else None
                response = self.client.get("/api/v1/health")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {
                    "status": "ok",
                    "model_loaded": lstm,
                    "models": {"lstm": lstm, "baseline": baseline},
                })

    def test_missing_checkpoints_return_service_unavailable(self):
        for model in ("lstm", "baseline"):
            with self.subTest(model=model):
                response = self.client.post("/api/v1/predict", json={
                    "review": "The acting was wonderful.", "model": model,
                })
                self.assertEqual(response.status_code, 503)
                self.assertIn("checkpoint", response.json()["detail"].lower())
        self.assertEqual(self.client.get("/api/v1/model").status_code, 503)

    def test_lstm_is_default_when_model_is_omitted(self):
        response = self.client.post("/api/v1/predict", json={"review": "Great acting."})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Model checkpoint is unavailable")

    def test_baseline_prediction_returns_labelled_probabilities(self):
        class FakeBaseline:
            classes_ = [0, 1, 2]

            def predict_proba(self, texts):
                self.texts = texts
                return numpy.array([[0.1, 0.2, 0.7]])

        app.state.baseline = FakeBaseline()
        response = self.client.post("/api/v1/predict", json={
            "review": "  Great acting.  ", "model": "baseline",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(app.state.baseline.texts, ["Great acting."])
        body = response.json()
        self.assertEqual(body["sentiment"], "positive")
        self.assertEqual(body["model_version"], "sst5_tfidf_logreg_v2")
        for label, score in {"negative": 0.1, "neutral": 0.2, "positive": 0.7}.items():
            self.assertAlmostEqual(body["probabilities"][label], score)

    def test_explain_returns_word_impacts(self):
        class FakeBaseline:
            classes_ = [0, 1, 2]

            def predict_proba(self, texts):
                # "fabulous" makes a review more positive; every other word is neutral.
                return numpy.array([[0.1, 0.2, 0.7] if "fabulous" in text else [0.4, 0.4, 0.2] for text in texts])

        app.state.baseline = FakeBaseline()
        response = self.client.post("/api/v1/predict", json={
            "review": "the movie was fabulous", "model": "baseline", "explain": True,
        })
        self.assertEqual(response.status_code, 200)
        words = {item["text"]: item["impact"] for item in response.json()["words"]}
        self.assertEqual(list(words), ["the", "movie", "was", "fabulous"])
        self.assertAlmostEqual(words["fabulous"], 2.639, places=2)
        self.assertAlmostEqual(words["movie"], 0.0)

    def test_batch_predicts_every_review_in_order(self):
        class FakeBaseline:
            classes_ = [0, 1, 2]

            def predict_proba(self, texts):
                return numpy.array([[0.1, 0.1, 0.8] if "great" in text else [0.8, 0.1, 0.1] for text in texts])

        app.state.baseline = FakeBaseline()
        response = self.client.post("/api/v1/predict-batch", json={
            "reviews": ["  great film ", "dull film"], "model": "baseline",
        })
        self.assertEqual(response.status_code, 200)
        results = response.json()["results"]
        self.assertEqual([item["review"] for item in results], ["great film", "dull film"])
        self.assertEqual([item["sentiment"] for item in results], ["positive", "negative"])

    def test_batch_rejects_empty_blank_and_oversized_lists(self):
        app.state.baseline = object()
        for body in ({"reviews": []}, {"reviews": ["fine", "  "]}, {"reviews": ["ok"] * 201}):
            with self.subTest(size=len(body["reviews"])):
                self.assertIn(self.client.post("/api/v1/predict-batch", json=body).status_code, (422,))

    def test_examples_route_serves_file_or_explains_why_not(self):
        app.state.examples = None
        self.assertEqual(self.client.get("/api/v1/examples").status_code, 404)
        app.state.examples = {"samples": [{"text": "Fine.", "label": "neutral"}], "failures": []}
        self.assertEqual(self.client.get("/api/v1/examples").json()["samples"][0]["label"], "neutral")

    def test_invalid_input_returns_validation_error(self):
        for body in ({}, {"review": ""}, {"review": " \n\t "},
                     {"review": 42}, {"review": "a" * 10001},
                     {"review": "Great acting.", "model": "unknown"}):
            with self.subTest(body_type=list(body)):
                self.assertEqual(self.client.post("/api/v1/predict", json=body).status_code, 422)


if __name__ == "__main__":
    unittest.main()
