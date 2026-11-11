import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

function ModelResults() {
  const [model, setModel] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/v1/model', { signal: controller.signal })
      .then(response => {
        if (!response.ok) throw new Error('Model details unavailable');
        return response.json();
      })
      .then(setModel)
      .catch(() => {
        if (!controller.signal.aborted) setFailed(true);
      });
    return () => controller.abort();
  }, []);

  const metrics = Object.entries(model?.test_metrics ?? {});
  return (
    <section className="dashboard" aria-labelledby="performance">
      <h2 id="performance">Model performance</h2>
      {failed && <p role="status">Model results are unavailable. Check that FastAPI is running with a checkpoint.</p>}
      {!failed && !model && <p role="status">Loading test results…</p>}
      {model && <>
        <p>Held-out test results saved with model <strong>{model.model_version}</strong>.</p>
        {metrics.length ? <div className="metrics">
          <table>
            <caption>Test scores by model</caption>
            <thead><tr><th scope="col">Model</th><th scope="col">Accuracy</th><th scope="col">Macro F1</th></tr></thead>
            <tbody>
              {metrics.map(([name, scores]) => (
                <tr key={name}>
                  <th scope="row">{name}</th>
                  <td>{(scores.accuracy * 100).toFixed(1)}%</td>
                  <td>{scores.macro_f1.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div> : <p>This checkpoint has no saved test scores.</p>}
      </>}
    </section>
  );
}

function App() {
  const [review, setReview] = useState('');
  const [selectedModel, setSelectedModel] = useState('lstm');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [availability, setAvailability] = useState(null);
  const [availabilityError, setAvailabilityError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/v1/health', { signal: controller.signal })
      .then(response => {
        if (!response.ok) throw new Error('Health check failed');
        return response.json();
      })
      .then(data => {
        if (typeof data.models?.lstm !== 'boolean' || typeof data.models?.baseline !== 'boolean') {
          throw new Error('Model availability missing');
        }
        setAvailability(data.models);
        if (!data.models.lstm && data.models.baseline) setSelectedModel('baseline');
      })
      .catch(() => {
        if (!controller.signal.aborted) setAvailabilityError('Could not check model availability. Start FastAPI and reload this page.');
      });
    return () => controller.abort();
  }, []);

  async function analyze(event) {
    event.preventDefault();
    setBusy(true);
    setResult(null);
    setError('');
    try {
      const response = await fetch('/api/v1/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review: review.trim(), model: selectedModel }),
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(typeof data.detail === 'string' ? data.detail : 'Prediction is unavailable. Check that FastAPI is running.');
      }
      setResult(await response.json());
    } catch (failure) {
      setError(failure instanceof TypeError ? 'Could not reach FastAPI. Check that it is running.' : failure.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <header>
        <p className="eyebrow">MOVIE REVIEW CLASSIFIER</p>
        <h1>Read the sentiment</h1>
        <p>Enter a movie review to see how the trained model classifies it.</p>
      </header>
      <form onSubmit={analyze}>
        <label htmlFor="review">Movie review</label>
        <textarea id="review" rows={6} maxLength={10000} required disabled={busy}
          placeholder="The acting was wonderful." value={review} onChange={event => setReview(event.target.value)} />
        <label htmlFor="prediction-model">Prediction model</label>
        <select id="prediction-model" disabled={busy || !availability} value={selectedModel} onChange={event => {
          setSelectedModel(event.target.value);
          setResult(null);
          setError('');
        }}>
          <option value="lstm" disabled={!availability?.lstm}>PyTorch LSTM</option>
          <option value="baseline" disabled={!availability?.baseline}>TF-IDF + Logistic Regression</option>
        </select>
        {!availability && !availabilityError && <p role="status">Checking available models…</p>}
        {availabilityError && <p role="alert">{availabilityError}</p>}
        {availability && !availability.lstm && <p role="status">LSTM unavailable. Add sst5_lstm_v1.pt to the backend and restart FastAPI.</p>}
        {availability && !availability.baseline && <p role="status">TF-IDF unavailable. Run backend/train_baseline.py and restart FastAPI.</p>}
        <button disabled={busy || !review.trim() || !availability?.[selectedModel]}>{busy ? 'Analyzing…' : 'Analyze review'}</button>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
      <section aria-live="polite">
        {result && <>
          <h2>Prediction <span>{result.sentiment}</span></h2>
          <dl>
            {Object.entries(result.probabilities).map(([label, score]) => (
              <div key={label}><dt>{label}</dt><dd>{(score * 100).toFixed(1)}%</dd></div>
            ))}
          </dl>
          <small>Model {result.model_version}</small>
        </>}
      </section>
      <ModelResults />
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
