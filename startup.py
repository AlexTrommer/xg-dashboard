from pathlib import Path

def ensure_data():
    required = [
        "data/shots.parquet",
        "data/teams.parquet",
        "data/players.parquet",
        "data/matches.parquet",
        "data/timeline.parquet",
        "data/watchlist.parquet",
    ]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        print(f"[startup] WARNING: Missing data files: {missing}")
        print("[startup] Skipping fetch — run data.py locally and push updated parquets.")
    else:
        print("[startup] Data files present ✓")
