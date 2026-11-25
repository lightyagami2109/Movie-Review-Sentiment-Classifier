import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { initializeApp } from 'firebase/app';
import {
  createUserWithEmailAndPassword, getAuth, onAuthStateChanged, sendPasswordResetEmail,
  signInWithEmailAndPassword, signOut as firebaseSignOut,
} from 'firebase/auth';
import { setTokenProvider } from './session.js';
import './login.css';

// These Firebase settings are public by design (they ship in every Firebase web page).
// Locally they come from the project's .env file; when deployed, from Space variables.
// Without them the site still works as a guest-only app.
const env = import.meta.env;
const FIREBASE = {
  apiKey: (env.VITE_FIREBASE_API_KEY || '').trim(),
  projectId: (env.VITE_FIREBASE_PROJECT_ID || '').trim(),
  appId: (env.VITE_FIREBASE_APP_ID || '').trim(),
  authDomain: (env.VITE_FIREBASE_AUTH_DOMAIN || '').trim(),
};
if (!FIREBASE.authDomain && FIREBASE.projectId) FIREBASE.authDomain = `${FIREBASE.projectId}.firebaseapp.com`;
const CONFIGURED = Boolean(FIREBASE.apiKey && FIREBASE.projectId);
const auth = CONFIGURED ? getAuth(initializeApp(FIREBASE)) : null;
const GUEST_KEY = 'guest';

function readSession(key) {
  try { return sessionStorage.getItem(key); } catch { return null; }
}
function writeSession(key, value) {
  try {
    if (value === null) sessionStorage.removeItem(key);
    else sessionStorage.setItem(key, value);
  } catch { /* storage can be blocked; guest mode just won't survive a reload */ }
}

/* ---------- Account state shared with the rest of the app ---------- */

const AccountContext = createContext(null);
export const useAccount = () => useContext(AccountContext);

const NOT_CONFIGURED = {
  configured: false, ready: true, signedIn: false, guest: true, email: '',
  continueAsGuest() {}, requestLogin() {}, async signOut() {},
};

function FirebaseBridge({ guest, setGuest, children }) {
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(false);

  useEffect(() => onAuthStateChanged(auth, next => { setUser(next); setReady(true); }), []);

  useEffect(() => {
    // getIdToken() hands back a fresh token automatically when the old one is about to expire.
    setTokenProvider(async () => {
      try { return auth.currentUser ? await auth.currentUser.getIdToken() : null; } catch { return null; }
    });
    return () => setTokenProvider(null);
  }, []);

  useEffect(() => {
    if (user && guest) setGuest(false);
  }, [user, guest, setGuest]);

  const value = useMemo(() => ({
    configured: true,
    ready,
    signedIn: Boolean(user),
    guest: guest && !user,
    email: user?.email || '',
    continueAsGuest: () => setGuest(true),
    requestLogin: () => setGuest(false),
    signOut: async () => { setGuest(false); await firebaseSignOut(auth); },
  }), [ready, user, guest, setGuest]);

  return <AccountContext.Provider value={value}>{children}</AccountContext.Provider>;
}

export function AuthProvider({ children }) {
  const [guest, setGuestState] = useState(() => readSession(GUEST_KEY) === '1');
  const setGuest = useCallback(value => {
    setGuestState(value);
    writeSession(GUEST_KEY, value ? '1' : null);
  }, []);

  if (!CONFIGURED) {
    return <AccountContext.Provider value={NOT_CONFIGURED}>{children}</AccountContext.Provider>;
  }
  return <FirebaseBridge guest={guest} setGuest={setGuest}>{children}</FirebaseBridge>;
}

/* ---------- Poster collage (all art is original and drawn in code) ---------- */

const DARK = '#0b0f1e';
const NIGHT = '#05070f';

const SCENES = {
  city: (
    <g fill={DARK}>
      <rect x="0" y="244" width="200" height="56" />
      <rect x="6" y="204" width="24" height="60" />
      <rect x="34" y="222" width="16" height="42" />
      <rect x="54" y="180" width="28" height="84" />
      <rect x="86" y="212" width="18" height="52" />
      <rect x="108" y="194" width="26" height="70" />
      <rect x="138" y="226" width="20" height="38" />
      <rect x="162" y="186" width="32" height="78" />
    </g>
  ),
  mountains: (
    <g>
      <path d="M0 300V208l28-34 20 22 38-62 32 46 22-26 60 78V300z" fill={DARK} />
      <path d="M0 300V252l40-24 36 20 44-30 80 40V300z" fill={NIGHT} />
    </g>
  ),
  sea: (
    <g>
      <g fill={DARK}>
        <rect x="0" y="226" width="200" height="74" />
        <path d="M96 226l12-44 12 44z" />
        <path d="M84 226h48l-8 14H92z" />
      </g>
      <path d="M10 254h40M70 264h60M150 252h40" stroke="#ffffff" strokeOpacity=".16" strokeWidth="2" strokeLinecap="round" />
    </g>
  ),
  valley: (
    <g>
      <path d="M0 300V232c40-26 84-30 120-8 30 18 52 14 80-6V300z" fill={DARK} />
      <path d="M0 300V268c36-14 70-10 100 4 36 16 70 10 100-6V300z" fill={NIGHT} />
    </g>
  ),
};

const POSTERS = [
  { title: 'Ember Road', line: 'Every road ends somewhere', sky: ['#2b1d63', '#d0592b'], orb: '#e6d7a3', at: [64, 168], r: 24, scene: 'city', ink: '#f1e7d0' },
  { title: 'Neon Harbor', line: 'Nobody leaves the valley', sky: ['#1c1f2e', '#46384a'], orb: '#c9c3cf', at: [140, 96], r: 16, scene: 'sea', ink: '#e9e4ee' },
  { title: 'Paper Moons', line: 'The tide remembers', sky: ['#e3bd83', '#d9945a'], orb: '#f6edd3', at: [146, 124], r: 26, scene: 'city', ink: '#2a2118' },
  { title: 'Iron Lullaby', line: 'One crossing. No return.', sky: ['#0f4b57', '#1f6e72'], orb: '#d3dcdf', at: [58, 110], r: 22, scene: 'sea', ink: '#e6f0f0' },
  { title: 'Midnight Ferry', line: 'No signal. No return.', sky: ['#20242f', '#3d414c'], orb: '#d8d6d2', at: [112, 142], r: 26, scene: 'mountains', ink: '#e8e6e2' },
  { title: 'Copper Sky', line: 'Listen closely', sky: ['#241d5e', '#c2562f'], orb: '#f0d9a0', at: [128, 150], r: 20, scene: 'city', ink: '#f2e8d2' },
  { title: 'Glass Orchard', line: 'Every road ends somewhere', sky: ['#14442d', '#3d7d4c'], orb: '#ebe4ba', at: [52, 140], r: 22, scene: 'valley', ink: '#eef2de' },
  { title: 'Salt & Static', line: 'In cinemas Friday', sky: ['#3b1d3a', '#b5494b'], orb: '#f2d19c', at: [138, 136], r: 24, scene: 'mountains', ink: '#f6e6d4' },
  { title: 'The Long Thaw', line: 'Some winters stay', sky: ['#1d3b5a', '#7ea6b8'], orb: '#f4f1e6', at: [70, 120], r: 18, scene: 'mountains', ink: '#f4f1e6' },
  { title: 'Lantern Hill', line: 'Follow the light', sky: ['#3a1f4d', '#e08a3c'], orb: '#ffe7a8', at: [132, 126], r: 22, scene: 'valley', ink: '#fbeed2' },
  { title: 'Quiet Machines', line: 'They are listening', sky: ['#10202c', '#2f4c5f'], orb: '#bfe0e6', at: [60, 130], r: 18, scene: 'city', ink: '#e1f0f3' },
  { title: 'Blue Hour', line: 'In cinemas Friday', sky: ['#14214a', '#5a6fb0'], orb: '#e9e1c9', at: [120, 112], r: 22, scene: 'sea', ink: '#e8ecf8' },
];

// Real output of the website's TF-IDF + Logistic Regression model (sst5_tfidf_logreg_v2)
// for these original sentences. Highlighted words are the ones that moved the score most.
const SAMPLE_REVIEWS = [
  { text: 'Loved every minute, a joyful ride.', mark: ['Loved', 'joyful'], label: 'positive', score: '94%' },
  { text: 'The ending felt rushed and predictable.', mark: ['rushed', 'predictable'], label: 'negative', score: '97%' },
  { text: 'A gorgeous, heartfelt film that earns every tear.', mark: ['gorgeous', 'heartfelt'], label: 'positive', score: '>99%' },
  { text: 'Pretty to look at, hollow underneath.', mark: ['hollow'], label: 'negative', score: '46%' },
  { text: 'Sharp writing and a cast that clearly cares.', mark: ['Sharp'], label: 'positive', score: '85%' },
  { text: 'Slow, bloated and far too long.', mark: ['Slow', 'bloated'], label: 'negative', score: '>99%' },
];

const COLUMNS = 9;
const PER_COLUMN = 4;
const bare = token => token.replace(/[^\p{L}\p{N}]/gu, '').toLowerCase();
const sentence = label => label.charAt(0).toUpperCase() + label.slice(1);

function SampleReview({ review }) {
  const marks = review.mark.map(bare);
  return (
    <div className="poster-review">
      <p>
        {review.text.split(' ').map((token, index) => (
          <React.Fragment key={index}>
            {index > 0 && ' '}
            {marks.includes(bare(token)) ? <mark className={review.label}>{token}</mark> : token}
          </React.Fragment>
        ))}
      </p>
      <div>
        <span>Review Reader</span>
        <b className={`mini ${review.label}`}>{sentence(review.label)} {review.score}</b>
      </div>
    </div>
  );
}

function Poster({ poster, review }) {
  return (
    <div className="poster" style={{ background: `linear-gradient(180deg, ${poster.sky[0]}, ${poster.sky[1]})`, color: poster.ink }}>
      <svg className="poster-art" viewBox="0 0 200 300" preserveAspectRatio="xMidYMax slice">
        <circle cx={poster.at[0]} cy={poster.at[1]} r={poster.r} fill={poster.orb} opacity=".92" />
        {SCENES[poster.scene]}
      </svg>
      <p className="poster-line">{poster.line}</p>
      <p className="poster-title">{poster.title}</p>
      {review && <SampleReview review={review} />}
    </div>
  );
}

function Collage() {
  const columns = useMemo(() => Array.from({ length: COLUMNS }, (_, c) => (
    Array.from({ length: PER_COLUMN }, (_, r) => ({
      poster: POSTERS[(c * 5 + r * 3) % POSTERS.length],
      review: (c + r) % 3 === 1 ? SAMPLE_REVIEWS[(c * 2 + r) % SAMPLE_REVIEWS.length] : null,
    }))
  )), []);
  return (
    <div className="collage" aria-hidden="true">
      <div className="collage-grid">
        {columns.map((column, c) => (
          <div className="collage-col" key={c}>
            {column.map((item, r) => <Poster key={r} poster={item.poster} review={item.review} />)}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ---------- Sign-in page ---------- */

function explain(error) {
  switch (error?.code) {
    case 'auth/invalid-credential':
    case 'auth/wrong-password':
    case 'auth/user-not-found': return 'That email and password do not match. Check them or create an account.';
    case 'auth/invalid-email': return 'Enter a valid email address.';
    case 'auth/email-already-in-use': return 'An account with this email already exists. Use Sign in instead.';
    case 'auth/weak-password': return 'Choose a password with at least 6 characters.';
    case 'auth/too-many-requests': return 'Too many attempts. Wait a few minutes and try again.';
    case 'auth/network-request-failed': return 'Could not reach the sign-in service. Check your connection.';
    case 'auth/user-disabled': return 'This account has been disabled.';
    case 'auth/operation-not-allowed': return 'Email sign-in is not turned on for this Firebase project yet.';
    default: return 'Something went wrong. Try again.';
  }
}

function AuthForm({ mode }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [info, setInfo] = useState('');
  const signup = mode === 'signup';

  async function submit(event) {
    event.preventDefault();
    setBusy(true); setError(''); setInfo('');
    try {
      const address = email.trim();
      if (signup) await createUserWithEmailAndPassword(auth, address, password);
      else await signInWithEmailAndPassword(auth, address, password);
      // On success the page switches to the app by itself.
    } catch (failure) {
      setError(explain(failure));
      setBusy(false);
    }
  }

  async function reset() {
    setError(''); setInfo('');
    if (!email.trim()) { setError('Type your email above first, then choose Forgot password.'); return; }
    try {
      await sendPasswordResetEmail(auth, email.trim());
      setInfo('If an account exists for that email, a reset link is on its way.');
    } catch (failure) {
      setError(explain(failure));
    }
  }

  return (
    <form className="auth-fields" onSubmit={submit} noValidate>
      <div>
        <label htmlFor="auth-email">Email</label>
        <input id="auth-email" type="email" autoComplete="email" inputMode="email" required
          value={email} onChange={event => setEmail(event.target.value)} />
      </div>
      <div>
        <label htmlFor="auth-password">Password</label>
        <div className="auth-pass">
          <input id="auth-password" type={show ? 'text' : 'password'} required minLength={6}
            autoComplete={signup ? 'new-password' : 'current-password'}
            aria-describedby={signup ? 'auth-hint' : undefined}
            value={password} onChange={event => setPassword(event.target.value)} />
          <button type="button" className="auth-eye" onClick={() => setShow(value => !value)}
            aria-label={show ? 'Hide password' : 'Show password'} aria-pressed={show}>
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8"
              strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z" />
              <circle cx="12" cy="12" r="3" />
              {show && <path d="M4 4l16 16" />}
            </svg>
          </button>
        </div>
        {signup && <p className="auth-hint" id="auth-hint">At least 6 characters.</p>}
      </div>
      {error && <p className="error auth-error" role="alert">{error}</p>}
      {info && <p className="note auth-info" role="status">{info}</p>}
      <button type="submit" className="primary" disabled={busy || !email.trim() || !password}>
        {busy ? 'One moment…' : signup ? 'Create account' : 'Sign in'}
      </button>
      {!signup && <button type="button" className="auth-link" onClick={reset}>Forgot password?</button>}
    </form>
  );
}

const MODES = [
  { key: 'signin', label: 'Sign in', title: 'Sign in to start reading reviews' },
  { key: 'signup', label: 'Create account', title: 'Create your account' },
];

export function LoginPage({ theme, onToggleTheme, onGuest, repoUrl }) {
  const [mode, setMode] = useState('signin');
  const current = MODES.find(item => item.key === mode);

  function choose(next) {
    if (next === mode) return;
    setMode(next);
  }
  function onTabKey(event) {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    event.currentTarget.querySelector('[aria-selected="false"]')?.focus();
    choose(mode === 'signin' ? 'signup' : 'signin');
  }

  return (
    <div className="auth-page">
      <Collage />
      <header className="top auth-top">
        <a className="brand" href="#top"><span className="logo" aria-hidden="true" />Review Reader</a>
        <nav aria-label="Main">
          <a href={repoUrl} target="_blank" rel="noreferrer">GitHub</a>
          <button type="button" className="theme" onClick={onToggleTheme}>{theme === 'dark' ? 'Light theme' : 'Dark theme'}</button>
        </nav>
      </header>
      <main className="auth-main" id="top">
        <section className="auth-card" aria-labelledby="auth-title">
          <p className="auth-kicker">Welcome to Review Reader</p>
          <h1 id="auth-title">{current.title}</h1>
          <p className="auth-lede">Paste a movie review and see if it is positive, neutral or negative, and which words moved the answer.</p>
          <div className="auth-tabs" role="tablist" aria-label="Account" onKeyDown={onTabKey}>
            {MODES.map(item => (
              <button key={item.key} type="button" role="tab" id={`tab-${item.key}`} aria-selected={mode === item.key}
                aria-controls="auth-panel" tabIndex={mode === item.key ? 0 : -1} onClick={() => choose(item.key)}>
                {item.label}
              </button>
            ))}
          </div>
          <div className="auth-form" id="auth-panel" role="tabpanel" aria-labelledby={`tab-${mode}`}>
            <AuthForm key={mode} mode={mode} />
          </div>
          <button type="button" className="auth-guest" onClick={onGuest}>Continue as guest</button>
          <p className="auth-note">Guests can analyze one review at a time. Sign in to analyze many reviews at once.</p>
          <p className="auth-foot">Trained on the Stanford Sentiment Treebank, 3 classes.</p>
        </section>
      </main>
    </div>
  );
}
