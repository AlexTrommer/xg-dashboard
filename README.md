# xG Dashboard

XGBoost-powered expected goals dashboard for the top 5 European leagues,
built on Understat data with possession-level team xG.

## Local setup

```bash
pip install -r requirements.txt

# Pull data and train model (takes ~10–20 min first run)
python data.py

# Start the server
uvicorn api:app --reload --port 8000
# Open http://localhost:8000
```

## Deploy to Railway

1. Push this folder to a GitHub repo
2. Go to railway.app → New Project → Deploy from GitHub repo
3. Railway auto-detects the Procfile and deploys

**Important:** Railway's free tier has an ephemeral filesystem — data fetched
at runtime won't persist across restarts. For a persistent deployment either:
- Add a Railway Volume (persistent disk) mounted at `/app/data`
- Or use Railway's scheduled job to re-fetch on startup via a startup script

### Startup script with auto-fetch (recommended for Railway)

Add this to `railway.toml` to fetch data on first boot if not present:

```toml
[deploy]
startCommand = "python -c \"from pathlib import Path; Path('data/shots.parquet').exists() or __import__('data').refresh(['2024'])\" && uvicorn api:app --host 0.0.0.0 --port $PORT"
```

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /api/status` | Last update time |
| `GET /api/teams?league=&season=` | Team xG table |
| `GET /api/players?league=&team=&season=&min_shots=` | Player xG table |
| `GET /api/matches?league=&team=&season=` | Recent matches |
| `GET /api/timeline/{match_id}` | Per-minute cumulative xG for a match |
| `GET /api/shots?league=&team=&player=&season=` | Raw shot data for shot map |
| `GET /api/refresh` | Trigger manual data refresh |

## Leagues

- Premier League (`EPL`)
- La Liga (`La_liga`)
- Bundesliga (`Bundesliga`)
- Serie A (`Serie_A`)
- Ligue 1 (`Ligue_1`)

## Architecture

```
data.py      Understat fetch → feature engineering → XGBoost train → parquet files
api.py       FastAPI — serves parquet data as JSON + schedules 6-hour refresh
static/      Single-page frontend (Chart.js, vanilla JS)
```

## Notes

- Data refreshes automatically every 6 hours (current season only)
- Model trained on all available seasons; penalties fixed at 0.76 xG
- Team xG uses possession-level formula: 1 − ∏(1 − xGᵢ) per attack
- Player xG is additive (each shot is independent)
