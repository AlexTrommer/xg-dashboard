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
        print(f"[startup] Missing data files: {missing}")
        print("[startup] Running data.py to fetch and train…")
        import data as D
        D.refresh()
        print("[startup] Done ✓")
    else:
        print("[startup] Data files present — skipping initial fetch")
