# xG Dashboard

XGBoost-powered expected goals dashboard for the top 5 European football leagues, built on Understat shot data. Features per-situation specialist models, a rolling form guide (Watch List), interactive shot maps, and cumulative xG match timelines.

Live dashboard: (https://xg-dashboard.up.railway.app/)

## Model

Three specialist XGBoost models are trained on ~250k shots (2020–2024) and evaluated on a 2025 holdout season:

| Metric | Ours | Understat |
|---|---|---|
| ROC-AUC | 0.792 | 0.805 |
| Brier score | 0.074 | 0.072 |

Each specialist uses a tailored feature set and independent GridSearchCV (3-fold CV, Brier score):

- **OpenPlay** — full 24-feature set including geometry, shot type flags, interaction terms, and a `fast_break` proxy for transitions (Understat doesn't tag counter-attacks natively)
- **FromCorner** — 13 features focused on header mechanics: header × distance, header × angle, centrality, weak-angle header flag
- **SetPiece** — covers indirect free kicks and direct free kicks; 13 features focused on distance, angle and shot type

All specialists are wrapped in `CalibratedClassifierCV(method="isotonic", cv=5)`. Penalties are assigned a fixed xG of **0.76** rather than modelled.

## Features (24 global)

| Group | Features |
|---|---|
| Geometry | distance, angle_deg, distance², distance×angle, distance×angle² |
| Shot type | is_header, is_set_piece, is_rebound, is_cross, is_throughball |
| Interactions | header×dist, header×angle, cross×dist, throughball×dist, rebound×dist |
| Zone / context | shot_zone, is_big_chance, minute_norm, central_threat, is_penalty_area |
| Proxy | fast_break, assisted_header, is_cutback, weak_angle_header |

## Local setup

```bash
pip install -r requirements.txt

# Optional: sanity check before the full run (~2 min)
python check.py

# Fetch all data and train model (~15–20 min first run)
python data.py

python train.py           # full grid
python train.py --quick   # faster, smaller grid

# Rebuild aggregated tables from existing shots (skip re-fetch)
python data.py --rebuild-tables

# Start the server
uvicorn api:app --reload --port 8000
# Open http://localhost:8000
```

Data refreshes automatically every **24 hours** (current season only).

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /api/status` | Last update time |
| `GET /api/leagues` | Available leagues |
| `GET /api/seasons` | Available seasons |
| `GET /api/teams?league=&season=` | Team xG table |
| `GET /api/players?league=&team=&season=&min_shots=&limit=` | Player xG rankings |
| `GET /api/matches?league=&team=&season=&limit=` | Recent matches |
| `GET /api/timeline/{match_id}` | Per-minute cumulative xG for a match |
| `GET /api/shots?league=&team=&player=&season=&match_id=&limit=` | Raw shot data |
| `GET /api/watchlist/form?league=&limit=` | Rolling form guide (last 5 games) |
| `GET /api/refresh` | Trigger manual data refresh (background) |

## Leagues

| Key | Name |
|---|---|
| `EPL` | Premier League |
| `La_liga` | La Liga |
| `Bundesliga` | Bundesliga |
| `Serie_A` | Serie A |
| `Ligue_1` | Ligue 1 |

## Architecture

```
data.py       Understat fetch → coordinate filter → feature engineering →
              XGBoost train → calibration → parquet aggregation
train.py      Per-situation specialist training with full GridSearchCV
              and holdout evaluation (2025 season)
check.py      Pre-flight sanity check — imports, Understat connectivity,
              feature engineering, model train/predict (~2 min)
api.py        FastAPI — serves parquet data as JSON, schedules 24-hour
              refresh of current season, mounts static frontend
startup.py    First-boot data fetch — runs data.py in background thread
              if parquet files are missing
static/       Single-page frontend (Chart.js, vanilla JS)
data/         Parquet files + trained model (gitignored)
```

## Notes

- Out-of-range shots are filtered before training: x < 17 or x > 101, y < 2 or y > 98 (penalties exempt)
- Team xG uses possession-level formula: 1 − ∏(1 − xGᵢ) per attacking sequence, in addition to standard additive xG
- Player xG is additive (each shot treated independently)
- Watch List rates players 0–100 on npxG/90 (60%) and xG/shot (40%) over their last 5 games — no outcome dependency
- Understat doesn't expose freeze-frame data (defender positions), which is the main reason AUC trails Understat's own model
- Counter-attack context approximated via `fast_break` flag (last action = TakeOn, BallRecovery, or Throughball) since Understat tags counters as OpenPlay
