import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';
import { AuthProvider, LoginPage, SiteBackdrop, useAccount } from './auth.jsx';
import { getAuthToken } from './session.js';
import { useTheme } from './theme.js';

const MODELS = {
  baseline: { name: 'TF-IDF + Logistic Regression', hint: 'Fast and the most accurate here' },
  lstm: { name: 'PyTorch LSTM', hint: 'Reads word order' },
};
const LABELS = ['positive', 'neutral', 'negative'];
const EXAMPLES = [
  'I loved every minute of it.',
  'Boring, far too long, and the ending made no sense.',
  'It was okay, nothing special.',
  'Not bad at all, the acting was good.',
];
const REPO_URL = 'https://github.com/lightyagami2109/Movie-Review-Sentiment-Classifier';
const STRONG = 0.3; // log-odds shift needed before a word is highlighted
const HISTORY_SIZE = 5;
const MAX_BATCH = 200;
const SLOW_MS = 4000;

const percent = value => `${(value * 100).toFixed(1)}%`;
const title = text => text.charAt(0).toUpperCase() + text.slice(1);
const topScore = result => Math.max(...Object.values(result.probabilities));
const clean = word => word.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '').toLowerCase();
const shorten = (text, length = 90) => (text.length > length ? `${text.slice(0, length - 1)}…` : text);

function readStorage(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value === null ? fallback : value;
  } catch {
    return fallback;
  }
}
function writeStorage(key, value) {
  try { localStorage.setItem(key, value); } catch { /* storage can be blocked; the choice just won't persist */ }
}

async function postJson(url, body, fallbackMessage) {
  const headers = { 'Content-Type': 'application/json' };
  const token = await getAuthToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body) });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(typeof data.detail === 'string' ? data.detail : fallbackMessage);
  }
  return response.json();
}
const requestPrediction = (review, model, explain) =>
  postJson('/api/v1/predict', { review, model, explain }, 'The model could not answer. Check that the backend is running.');

function explainFailure(failure) {
  return failure instanceof TypeError ? 'Cannot reach the backend. Start it, then try again.' : failure.message;
}

function describeConfidence(result) {
  const sorted = Object.values(result.probabilities).sort((a, b) => b - a);
  const top = sorted[0];
  const gap = top - sorted[1];
  if (top >= 0.75) return { level: 'High', text: 'The model is very sure.' };
  if (gap >= 0.2) return { level: 'Medium', text: 'The model is fairly sure.' };
  return { level: 'Low', text: 'Close call. Treat this answer with care.' };
}

function keyWords(result) {
  if (!result.words || result.sentiment === 'neutral') return [];
  const direction = result.sentiment === 'positive' ? 1 : -1;
  const seen = new Set();
  return [...result.words]
    .filter(word => word.impact * direction > STRONG)
    .sort((a, b) => b.impact * direction - a.impact * direction)
    .map(word => clean(word.text))
    .filter(word => word && !seen.has(word) && seen.add(word))
    .slice(0, 2);
}

/* ---------- Result ---------- */

function Ticket({ result, modelName, truth }) {
  const confidence = describeConfidence(result);
  const drivers = keyWords(result);
  return (
    <section className={`ticket ${result.sentiment}`} aria-labelledby="prediction-title">
      <div className="ticket-main">
        <p className="small" id="prediction-title">Prediction</p>
        <p className="verdict">{title(result.sentiment)}</p>
        <p className="note">
          {confidence.text}
          {drivers.length > 0 && <> Most of the push comes from {drivers.map((word, index) => <React.Fragment key={word}>{index > 0 && ' and '}<q>{word}</q></React.Fragment>)}.</>}
        </p>
        {truth && (
          <p className={`truth ${truth === result.sentiment ? 'hit' : 'miss'}`}>
            <b>{truth === result.sentiment ? 'Correct.' : 'Missed.'}</b> The dataset labels this review {truth}.
          </p>
        )}
        {LABELS.map(label => (
          <div className="bar" key={label}>
            <span>{title(label)}</span>
            <div className="track" aria-hidden="true"><div className={`fill ${label}`} style={{ width: `${(result.probabilities[label] ?? 0) * 100}%` }} /></div>
            <b>{percent(result.probabilities[label] ?? 0)}</b>
          </div>
        ))}
      </div>
      <div className="ticket-stub">
        <p className="small">Confidence</p>
        <p className="big">{Math.round(topScore(result) * 100)}%</p>
        <p className="small">{confidence.level}<br />{modelName}</p>
      </div>
    </section>
  );
}

function TicketSkeleton({ slow }) {
  return (
    <section className="ticket skeleton-card" aria-busy="true" aria-label="Analyzing">
      <div className="ticket-main">
        <div className="skeleton" style={{ width: '5rem', height: '.8rem' }} />
        <div className="skeleton" style={{ width: '12rem', height: '3rem', margin: '.6rem 0 1rem' }} />
        <div className="skeleton" style={{ width: '90%', height: '.9rem', marginBottom: '1.25rem' }} />
        {[0, 1, 2].map(index => <div className="skeleton" key={index} style={{ height: '.8rem', margin: '1rem 0' }} />)}
        {slow && <p className="note" role="status">Still working. If the server was idle, it can take up to a minute to wake up.</p>}
      </div>
      <div className="ticket-stub"><div className="skeleton" style={{ width: '6rem', height: '2.5rem' }} /></div>
    </section>
  );
}

function Words({ result }) {
  if (!result.words?.length) return null;
  return (
    <section className="panel" aria-labelledby="words-title">
      <h2 id="words-title">Words that moved the score</h2>
      <p className="words">
        {result.words.map((word, index) => {
          const kind = word.impact > STRONG ? 'up' : word.impact < -STRONG ? 'down' : 'flat';
          return <React.Fragment key={index}><span className={`word ${kind}`} title={kind === 'flat' ? 'Little effect' : kind === 'up' ? 'Pushes toward positive' : 'Pushes toward negative'}>{word.text}</span>{' '}</React.Fragment>;
        })}
      </p>
      <p className="legend">
        <span><i className="swatch up" />Pushes toward positive</span>
        <span><i className="swatch down" />Pushes toward negative</span>
      </p>
    </section>
  );
}

function Agreement({ results }) {
  const rows = Object.keys(MODELS).map(key => ({ key, result: results[key] })).filter(row => row.result);
  if (rows.length < 2) return null;
  const agree = rows[0].result.sentiment === rows[1].result.sentiment;
  return (
    <section className="panel" aria-labelledby="agree-title">
      <h2 id="agree-title">Do the two models agree?</h2>
      <p className="note">{agree ? `Yes, both say ${rows[0].result.sentiment}.` : 'No. They read this review differently.'}</p>
      {rows.map(({ key, result }) => (
        <div className="compare-row" key={key}>
          <span>{MODELS[key].name}</span>
          <span className={`pill ${result.sentiment}`}>{title(result.sentiment)} {Math.round(topScore(result) * 100)}%</span>
        </div>
      ))}
    </section>
  );
}

/* ---------- History ---------- */

function History({ items, onUse, onClear }) {
  if (!items.length) return null;
  return (
    <section className="panel" aria-labelledby="history-title">
      <div className="panel-head">
        <h2 id="history-title">Recent reviews</h2>
        <button type="button" className="link" onClick={onClear}>Clear</button>
      </div>
      <ul className="plain">
        {items.map(item => (
          <li key={`${item.at}`}>
            <button type="button" className="history-item" onClick={() => onUse(item)}>
              <span className="clip">{shorten(item.text, 70)}</span>
              <span className={`pill ${item.sentiment}`}>{title(item.sentiment)} {Math.round(item.confidence * 100)}%</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

/* ---------- Where it fails ---------- */

function pickRandom(list, count) {
  const copy = [...list];
  for (let index = copy.length - 1; index > 0; index -= 1) {
    const other = Math.floor(Math.random() * (index + 1));
    [copy[index], copy[other]] = [copy[other], copy[index]];
  }
  return copy.slice(0, count);
}

function Failures({ failures, onTry, disabled }) {
  const [shown, setShown] = useState(() => pickRandom(failures, 4));
  if (!failures.length) return null;
  return (
    <section className="panel wide" aria-labelledby="fails-title">
      <div className="panel-head">
        <h2 id="fails-title">Where it fails</h2>
        <button type="button" className="link" onClick={() => setShown(pickRandom(failures, 4))}>Show others</button>
      </div>
      <p className="note">Real test reviews that the TF-IDF model got wrong while sounding sure. Many are neutral reviews with mixed feelings, which even people label differently.</p>
      <ul className="plain">
        {shown.map(item => (
          <li className="failure" key={item.text}>
            <p className="clip-3">{item.text}</p>
            <p className="meta">
              Dataset says <span className={`pill ${item.label}`}>{title(item.label)}</span> The model said <span className={`pill ${item.predicted}`}>{title(item.predicted)} {Math.round(item.confidence * 100)}%</span>
            </p>
            <button type="button" className="link" disabled={disabled} onClick={() => onTry(item)}>Try this review</button>
          </li>
        ))}
      </ul>
    </section>
  );
}

/* ---------- Batch mode ---------- */

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = '';
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') { cell += '"'; index += 1; }
      else if (char === '"') quoted = false;
      else cell += char;
    } else if (char === '"') quoted = true;
    else if (char === ',') { row.push(cell); cell = ''; }
    else if (char === '\n' || char === '\r') {
      if (char === '\r' && text[index + 1] === '\n') index += 1;
      row.push(cell); rows.push(row); row = []; cell = '';
    } else cell += char;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  return rows;
}

// One review per line. For CSV files, use the column named review/text/sentence, or the first column.
function extractReviews(text) {
  const rows = parseCsv(text).filter(row => row.some(cell => cell.trim()));
  if (!rows.length) return [];
  let column = 0;
  const header = rows[0].findIndex(cell => /^(review|text|sentence)$/i.test(cell.trim()));
  if (header >= 0) { column = header; rows.shift(); }
  return rows.map(row => (row[column] ?? '').trim()).filter(Boolean);
}

const safeCell = value => {
  const text = /^[=+\-@\t\r]/.test(value) ? `'${value}` : value; // stops spreadsheets running a review as a formula
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};

function downloadCsv(results) {
  const lines = ['review,prediction,positive,neutral,negative'];
  results.forEach(item => lines.push([
    safeCell(item.review), item.sentiment,
    item.probabilities.positive.toFixed(4), item.probabilities.neutral.toFixed(4), item.probabilities.negative.toFixed(4),
  ].join(',')));
  const url = URL.createObjectURL(new Blob([`${lines.join('\n')}\n`], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'review-predictions.csv';
  link.click();
  URL.revokeObjectURL(url);
}

function Batch({ modelKey, available, account }) {
  const [text, setText] = useState('');
  const [results, setResults] = useState(null);
  const [note, setNote] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function loadFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (file.size > 2_000_000) { setError('That file is too large. Use a file under 2 MB.'); return; }
    setError('');
    setText(await file.text());
    event.target.value = '';
  }

  async function run(event) {
    event.preventDefault();
    const all = extractReviews(text);
    if (!all.length) { setError('Add at least one review, one per line.'); return; }
    const reviews = all.slice(0, MAX_BATCH).map(review => review.slice(0, 10000));
    setBusy(true); setError(''); setResults(null);
    setNote(all.length > MAX_BATCH ? `Only the first ${MAX_BATCH} of ${all.length} reviews were analyzed.` : '');
    try {
      const data = await postJson('/api/v1/predict-batch', { reviews, model: modelKey }, 'The model could not answer. Check that the backend is running.');
      setResults(data.results);
    } catch (failure) {
      setError(explainFailure(failure));
    } finally {
      setBusy(false);
    }
  }

  if (!account.signedIn) {
    return (
      <section className="panel batch" id="batch" aria-labelledby="batch-title">
        <h2 id="batch-title">Analyze many reviews at once</h2>
        <p className="note">
          {account.configured
            ? `Sign in to paste or upload up to ${MAX_BATCH} reviews and download the predictions as a CSV file.`
            : 'Sign-in is not set up on this server, so analyzing many reviews at once is turned off.'}
        </p>
        {account.configured && <button type="button" className="primary fit" onClick={account.requestLogin}>Sign in</button>}
      </section>
    );
  }
  const counts = results ? LABELS.map(label => [label, results.filter(item => item.sentiment === label).length]) : [];
  return (
    <section className="panel batch" id="batch" aria-labelledby="batch-title">
      <h2 id="batch-title">Analyze many reviews at once</h2>
      <p className="note">Paste one review per line, or upload a .csv or .txt file (up to {MAX_BATCH} reviews). CSV files can use a column named review, text or sentence. Uses the model selected above: {MODELS[modelKey].name}.</p>
      <form onSubmit={run}>
        <label htmlFor="batch-text" className="visually-hidden">Reviews, one per line</label>
        <textarea id="batch-text" rows={6} value={text} disabled={busy} placeholder={'The acting was wonderful.\nA dull, predictable plot.\nIt was fine.'} onChange={event => setText(event.target.value)} />
        <div className="batch-actions">
          <button className="primary" disabled={busy || !text.trim() || !available}>{busy ? 'Analyzing…' : 'Analyze all'}</button>
          <label className="file">Upload a file<input type="file" accept=".csv,.txt,text/csv,text/plain" onChange={loadFile} disabled={busy} /></label>
        </div>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
      {results && (
        <div aria-live="polite">
          <div className="batch-summary">
            <p className="note">{note || `${results.length} reviews analyzed.`} {counts.map(([label, count]) => `${count} ${label}`).join(', ')}.</p>
            <button type="button" className="ghost" onClick={() => downloadCsv(results)}>Download CSV</button>
          </div>
          <div className="table-wrap">
            <table>
              <caption className="visually-hidden">Predictions for each review</caption>
              <thead><tr><th scope="col">Review</th><th scope="col">Prediction</th><th scope="col" className="num">Confidence</th></tr></thead>
              <tbody>
                {results.map((item, index) => (
                  <tr key={index}>
                    <th scope="row" className="review-cell" title={item.review}>{shorten(item.review, 110)}</th>
                    <td><span className={`pill ${item.sentiment}`}>{title(item.sentiment)}</span></td>
                    <td className="num">{Math.round(topScore(item) * 100)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}

/* ---------- Model results ---------- */

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

  const saved = Object.entries(model?.test_metrics ?? {});
  const lstm = saved.find(([name]) => /lstm/i.test(name));
  const tfidf = model?.baseline_metrics ?? saved.find(([name]) => /tf-?idf|baseline/i.test(name))?.[1];
  const rows = [];
  if (tfidf) rows.push({ name: MODELS.baseline.name, ...tfidf });
  if (lstm) rows.push({ name: MODELS.lstm.name, ...lstm[1] });
  const bestAccuracy = Math.max(...rows.map(row => row.accuracy));
  const bestF1 = Math.max(...rows.map(row => row.macro_f1));

  return (
    <section className="panel" id="results" aria-labelledby="performance">
      <h2 id="performance">Model results on unseen reviews</h2>
      {failed && <p className="note" role="status">Results are not available. Start the backend with a model file and reload this page.</p>}
      {!failed && !model && <p className="note" role="status">Loading results…</p>}
      {model && (rows.length ? <table>
        <caption className="visually-hidden">Test scores by model</caption>
        <thead><tr><th scope="col">Model</th><th scope="col" className="num">Accuracy</th><th scope="col" className="num">Macro F1</th></tr></thead>
        <tbody>
          {rows.map(row => (
            <tr key={row.name}>
              <th scope="row">{row.name}</th>
              <td className={`num ${row.accuracy === bestAccuracy ? 'best' : ''}`}>{(row.accuracy * 100).toFixed(1)}%</td>
              <td className={`num ${row.macro_f1 === bestF1 ? 'best' : ''}`}>{row.macro_f1.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table> : <p className="note">This model file has no saved test scores.</p>)}
    </section>
  );
}

/* ---------- App ---------- */

function loadHistory() {
  try {
    const items = JSON.parse(readStorage('history', '[]'));
    return Array.isArray(items) ? items.filter(item => item && typeof item.text === 'string').slice(0, HISTORY_SIZE) : [];
  } catch {
    return [];
  }
}

function App({ theme, onToggleTheme }) {
  const account = useAccount();
  const [review, setReview] = useState('');
  const [truth, setTruth] = useState(null);
  const [selectedModel, setSelectedModel] = useState('baseline');
  const [results, setResults] = useState(null);
  const [shownModel, setShownModel] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [slow, setSlow] = useState(false);
  const [availability, setAvailability] = useState(null);
  const [availabilityError, setAvailabilityError] = useState('');
  const [examples, setExamples] = useState(null);
  const [history, setHistory] = useState(loadHistory);
  const textarea = useRef(null);
  const lastRandom = useRef(-1);

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
        if (!data.models.baseline && data.models.lstm) setSelectedModel('lstm');
      })
      .catch(() => {
        if (!controller.signal.aborted) setAvailabilityError('Cannot reach the backend. Start it, then reload this page.');
      });
    fetch('/api/v1/examples', { signal: controller.signal })
      .then(response => (response.ok ? response.json() : null))
      .then(data => { if (data?.samples?.length) setExamples(data); })
      .catch(() => {});
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!busy && availability) { setSlow(false); return undefined; }
    const timer = setTimeout(() => setSlow(true), SLOW_MS);
    return () => clearTimeout(timer);
  }, [busy, availability]);

  function remember(text, result) {
    setHistory(previous => {
      const next = [{ text, sentiment: result.sentiment, confidence: topScore(result), at: Date.now() },
        ...previous.filter(item => item.text !== text)].slice(0, HISTORY_SIZE);
      writeStorage('history', JSON.stringify(next));
      return next;
    });
  }

  function clearHistory() {
    setHistory([]);
    writeStorage('history', '[]');
  }

  async function run(text, knownLabel = null, modelKey = selectedModel) {
    setBusy(true);
    setError('');
    const other = modelKey === 'baseline' ? 'lstm' : 'baseline';
    try {
      const [main, second] = await Promise.allSettled([
        requestPrediction(text, modelKey, true),
        availability?.[other] ? requestPrediction(text, other, false) : Promise.reject(new Error('unavailable')),
      ]);
      if (main.status === 'rejected') throw main.reason;
      setResults({ [modelKey]: main.value, ...(second.status === 'fulfilled' ? { [other]: second.value } : {}) });
      setShownModel(modelKey);
      setTruth(knownLabel);
      remember(text, main.value);
    } catch (failure) {
      setResults(null);
      setError(explainFailure(failure));
    } finally {
      setBusy(false);
    }
  }

  function analyze(event) {
    event.preventDefault();
    run(review.trim(), truth);
  }

  function useExample(text) {
    setReview(text);
    setTruth(null);
    setError('');
    textarea.current?.focus();
  }

  function tryDatasetReview(item, modelKey = selectedModel) {
    setReview(item.text);
    setSelectedModel(modelKey);
    run(item.text, item.label, modelKey);
  }

  function randomReview() {
    const samples = examples.samples;
    let index = Math.floor(Math.random() * samples.length);
    if (index === lastRandom.current) index = (index + 1) % samples.length;
    lastRandom.current = index;
    tryDatasetReview(samples[index]);
  }

  const shown = results && shownModel ? results[shownModel] : null;
  const ready = Boolean(availability?.[selectedModel]);
  const canAnalyze = !busy && review.trim() && ready;
  const failures = examples?.failures ?? [];

  return (
    <>
      <SiteBackdrop />
      <header className="top">
        <a className="brand" href="#top"><span className="logo" aria-hidden="true" />Review Reader</a>
        <nav aria-label="Main">
          <a href="#top" aria-current="page">Analyze</a>
          <a href="#batch">Batch</a>
          <a href="#results">Model results</a>
          <a href={REPO_URL} target="_blank" rel="noreferrer">GitHub</a>
          <button type="button" className="theme" onClick={onToggleTheme}>{theme === 'dark' ? 'Light theme' : 'Dark theme'}</button>
          {account.signedIn && <>
            <span className="account-email" title={account.email}>{account.email}</span>
            <button type="button" className="theme" onClick={account.signOut}>Sign out</button>
          </>}
          {!account.signedIn && account.configured && <button type="button" className="theme" onClick={account.requestLogin}>Sign in</button>}
        </nav>
      </header>
      <main id="top">
        <section className="hero">
          <h1>Is this movie review positive, neutral or negative?</h1>
          <p>Paste a review. Two trained models read it, show how sure they are, and point out the words that moved the answer.</p>
        </section>
        <div className="layout">
          <form className="panel" onSubmit={analyze}>
            <label htmlFor="review">Movie review</label>
            <textarea id="review" ref={textarea} rows={5} maxLength={10000} required disabled={busy}
              placeholder="The acting was wonderful." value={review}
              onChange={event => { setReview(event.target.value); setTruth(null); }} />
            <p className="count" aria-hidden="true">{review.length.toLocaleString()} / 10,000</p>

            <p className="label">Try an example</p>
            <div className="chips">
              {EXAMPLES.map(text => <button type="button" className="chip" key={text} disabled={busy} onClick={() => useExample(text)}>{text}</button>)}
              {examples && <button type="button" className="chip random" disabled={busy || !ready} onClick={randomReview}>Try a random review from the dataset</button>}
            </div>

            <fieldset disabled={busy || !availability}>
              <legend>Model</legend>
              <div className="segmented">
                {Object.entries(MODELS).map(([key, model]) => (
                  <label key={key} className={`${selectedModel === key ? 'on' : ''} ${availability && !availability[key] ? 'off' : ''}`}>
                    <input type="radio" name="model" value={key} checked={selectedModel === key} disabled={availability ? !availability[key] : false}
                      onChange={() => setSelectedModel(key)} />
                    <span>{model.name}</span>
                    <small>{availability && !availability[key] ? 'Model file missing' : model.hint}</small>
                  </label>
                ))}
              </div>
            </fieldset>
            {!availability && !availabilityError && <p className="note" role="status">{slow ? 'Still connecting. If the server was idle, it can take up to a minute to wake up.' : 'Checking the backend…'}</p>}
            {availabilityError && <p className="error" role="alert">{availabilityError}</p>}
            {availability && !availability.lstm && <p className="note" role="status">The LSTM needs sst5_lstm_v1.pt in the backend folder. Restart the backend after adding it.</p>}
            {availability && !availability.baseline && <p className="note" role="status">The TF-IDF model file is missing. Run python train_baseline.py in the backend folder.</p>}
            <button className="primary" disabled={!canAnalyze}>{busy ? 'Analyzing…' : 'Analyze review'}</button>
            {error && <p className="error" role="alert">{error}</p>}
          </form>

          <div className="stack" aria-live="polite">
            {busy ? <TicketSkeleton slow={slow} /> : shown ? <>
              <Ticket result={shown} modelName={MODELS[shownModel].name} truth={truth} />
              <Words result={shown} />
            </> : <section className="panel empty">
              <h2>Your result will appear here</h2>
              <p className="note">Type a review, pick an example, or try a random review from the dataset.</p>
            </section>}
            <History items={history} onClear={clearHistory} onUse={item => useExample(item.text)} />
          </div>
        </div>
        <div className="lower">
          {results && <Agreement results={results} />}
          <ModelResults />
          {failures.length > 0 && availability?.baseline && <Failures failures={failures} disabled={busy} onTry={item => tryDatasetReview(item, 'baseline')} />}
        </div>
        <div className="lower single">
          <Batch modelKey={selectedModel} available={ready} account={account} />
        </div>
      </main>
      <footer>
        <span>Trained on the Stanford Sentiment Treebank, grouped into 3 classes. About 1 in 3 reviews is still misjudged.</span>
        <span>Advanced Data Science project</span>
      </footer>
    </>
  );
}

/* ---------- Root: loading screen, sign-in page or the app ---------- */

const WAIT_MS = 8000;

function Root() {
  const account = useAccount();
  const [theme, toggleTheme] = useTheme();
  const [waited, setWaited] = useState(false);

  useEffect(() => {
    if (account.ready) return undefined;
    const timer = setTimeout(() => setWaited(true), WAIT_MS);
    return () => clearTimeout(timer);
  }, [account.ready]);

  if (!account.ready && !account.guest) {
    return (
      <main className="boot" role="status">
        <p>{waited ? 'Sign-in is taking longer than usual.' : 'Loading…'}</p>
        {waited && <button type="button" className="ghost" onClick={account.continueAsGuest}>Continue as guest</button>}
      </main>
    );
  }
  if (account.configured && !account.signedIn && !account.guest) {
    return <LoginPage theme={theme} onToggleTheme={toggleTheme} onGuest={account.continueAsGuest} repoUrl={REPO_URL} />;
  }
  return <App theme={theme} onToggleTheme={toggleTheme} />;
}

createRoot(document.getElementById('root')).render(<AuthProvider><Root /></AuthProvider>);
