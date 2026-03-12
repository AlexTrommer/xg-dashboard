#!/bin/bash
# run.sh — full xG dashboard workflow
# Usage:
#   ./run.sh            full run (fetch + full grid search + re-score + serve)
#   ./run.sh --quick    fast run (fetch + quick grid search + re-score + serve)
#   ./run.sh --serve    skip data/training, just start the server
#   ./run.sh --no-serve run data + training but don't start server at the end

set -e  # exit on any error

QUICK=""
SERVE=true
FETCH=true
TRAIN=true

for arg in "$@"; do
  case $arg in
    --quick)    QUICK="--quick" ;;
    --serve)    FETCH=false; TRAIN=false ;;
    --no-serve) SERVE=false ;;
  esac
done

# ── Colours ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'
BOLD='\033[1m'

log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${RESET} $1"; }
ok()   { echo -e "${GREEN}✓${RESET} $1"; }
warn() { echo -e "${YELLOW}⚠${RESET} $1"; }
err()  { echo -e "${RED}✗ $1${RESET}"; exit 1; }

echo ""
echo -e "${BOLD}================================================${RESET}"
echo -e "${BOLD}  xG Dashboard — automated workflow${RESET}"
echo -e "${BOLD}================================================${RESET}"
echo ""

START_TIME=$SECONDS

# ── Pre-flight check ───────────────────────────────────────────────────────────
log "Running pre-flight checks…"
python check.py || err "Pre-flight check failed — fix errors above before continuing"
echo ""

# ── Step 1: Fetch data ─────────────────────────────────────────────────────────
if [ "$FETCH" = true ]; then
  log "Step 1/3 — Fetching data from Understat (this takes ~20–40 min)…"
  python data.py || err "data.py failed"
  ok "Data fetched and parquets saved"
  echo ""
else
  warn "Skipping data fetch (--serve flag)"
fi

# ── Step 2: Train / grid search ────────────────────────────────────────────────
if [ "$TRAIN" = true ]; then
  if [ -n "$QUICK" ]; then
    log "Step 2/3 — Running QUICK grid search (~5–10 min)…"
  else
    log "Step 2/3 — Running FULL grid search (~1–3 hours)…"
    warn "Tip: run with --quick for a faster test pass"
  fi

  python train.py $QUICK || err "train.py failed"
  ok "Model trained and saved to data/model.pkl"
  echo ""

  # Step 2b: Re-score all shots with tuned model
  log "Step 2b/3 — Re-scoring all shots with tuned model…"
  python data.py || err "data.py re-score failed"
  ok "Parquets regenerated with tuned model predictions"
  echo ""
else
  warn "Skipping training (--serve flag)"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
ELAPSED=$(( SECONDS - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo -e "${BOLD}================================================${RESET}"
echo -e "${GREEN}${BOLD}  All steps complete in ${MINS}m ${SECS}s${RESET}"
echo -e "${BOLD}================================================${RESET}"
echo ""

# Print best params if available
if [ -f "data/best_params.json" ]; then
  ok "Best params found:"
  cat data/best_params.json
  echo ""
fi

# ── Step 3: Start server ───────────────────────────────────────────────────────
if [ "$SERVE" = true ]; then
  log "Step 3/3 — Starting server at http://localhost:8000"
  echo ""
  uvicorn api:app --reload --port 8000
else
  warn "Server not started (--no-serve flag)"
  log "To start the server run:  uvicorn api:app --reload --port 8000"
fi
