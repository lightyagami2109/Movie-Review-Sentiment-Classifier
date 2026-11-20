#!/usr/bin/env bash
# Starts the API (port 8000) and the website (port 5173) together.
# Run from the project folder with:  bash start.sh      Stop both with Ctrl+C.
set -e
cd "$(dirname "$0")"
ROOT="$PWD"

# 1. Find (or create) the Python virtual environment.
VENV=""
for folder in "$ROOT/backend/.venv" "$ROOT/.venv"; do
  if [ -f "$folder/bin/activate" ]; then VENV="$folder"; break; fi
done
if [ -z "$VENV" ]; then
  echo "Creating a Python environment in backend/.venv ..."
  python3 -m venv "$ROOT/backend/.venv"
  VENV="$ROOT/backend/.venv"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# 2. Make sure the packages match backend/requirements.txt (quick when nothing changed).
python -m pip install -q -r backend/requirements.txt || echo "Could not check packages (offline?). Continuing."
if [ ! -d frontend/node_modules ]; then (cd frontend && npm ci); fi

# 3. Warn about missing model files.
[ -f backend/sst5_lstm_v1.pt ] || echo "Note: backend/sst5_lstm_v1.pt is missing, so the LSTM option is off. Run the Colab notebook and download it."
[ -f backend/sst5_tfidf_logreg_v2.joblib ] || echo "Note: the TF-IDF model is missing. Run: cd backend && python train_baseline.py"

# 4. Free port 8000 if an old backend is still running from an earlier session.
if lsof -ti :8000 >/dev/null 2>&1; then
  echo "Port 8000 is busy. Stopping the old backend ..."
  lsof -ti :8000 | xargs kill
  sleep 1
fi

# 5. Start the API in the background and the website in front.
(cd backend && python -m uvicorn main:app --port 8000) &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT INT TERM
echo "Opening http://localhost:5173 ..."
(cd frontend && npm run dev -- --open)
