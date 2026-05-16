"""
api.py — FastAPI backend
"""

from pathlib import Path
from datetime import datetime
import gc
import threading

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from apscheduler.schedulers.background import BackgroundScheduler

import data as D
from startup import ensure_data

threading.Thread(target=ensure_data, daemon=True).start()

_CACHE: dict[str, pd.DataFrame] = {}
_CACHE_LOCK = threading.Lock()

AGGREGATE_TABLES = {"teams", "players", "matches", "timeline",
                    "watchlist", "watchlist_form"}


def _df(name: str) -> pd.DataFrame:
    """Load an aggregate parquet into cache on first access.
    shots.parquet is intentionally excluded — use _shots_query() instead."""
    if name not in AGGREGATE_TABLES:
        raise ValueError(f"Unknown aggregate table: {name}")
    with _CACHE_LOCK:
        if name not in _CACHE:
            p = Path(f"data/{name}.parquet")
            if not p.exists():
                raise HTTPException(503, f"Data not ready — run: python data.py  (missing: {name})")
            _CACHE[name] = pd.read_parquet(p)
        return _CACHE[name]


def _invalidate_cache(*names):
    """Evict specific tables (or all if none given)."""
    with _CACHE_LOCK:
        if names:
            for n in names:
                _CACHE.pop(n, None)
        else:
            _CACHE.clear()
    gc.collect()


def _shots_query(
    league: str | None = None,
    team: str | None = None,
    player: str | None = None,
    season: str | None = None,
    match_id: str | None = None,
    columns: list[str] | None = None,
    limit: int = 2000,
) -> pd.DataFrame:

    import pyarrow.parquet as pq

    p = Path("data/shots.parquet")
    if not p.exists():
        raise HTTPException(503, "Data not ready — run: python data.py")

    default_cols = [
        "match_id", "match_date", "player", "team", "opponent",
        "league", "season", "minute", "x", "y", "goal",
        "situation", "shot_type", "xg", "understat_xg",
    ]
    read_cols = columns or default_cols
    available = pq.read_schema(p).names
    read_cols = [c for c in read_cols if c in available]

    # Build pyarrow filter list
    filters = []
    if league: filters.append(("league", "=", league))
    if season: filters.append(("season", "=", season))
    if team: filters.append(("team",   "=", team))
    if player: filters.append(("player", "=", player))

    df = pq.read_table(
        p,
        columns=read_cols,
        filters=filters if filters else None,
    ).to_pandas()

    if match_id:
        df = df[df["match_id"].astype(str) == match_id]

    for c in (columns or default_cols):
        if c not in df.columns:
            df[c] = ""

    return df.head(limit)

app = FastAPI(title="xG Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

scheduler = BackgroundScheduler()

def _scheduled_refresh():
    D.refresh([D.CURRENT_SEASON])
    _invalidate_cache()  # clear all so next request picks up fresh data

scheduler.add_job(_scheduled_refresh, "interval", hours=24, id="auto_refresh")
scheduler.start()


def _last_updated():
    p = Path("data/last_updated.txt")
    return p.read_text().strip() if p.exists() else "never"


@app.get("/api/status")
def status():
    return {"last_updated": _last_updated(), "status": "ok"}


@app.get("/api/leagues")
def leagues():
    return list(D.LEAGUES.values())


@app.get("/api/seasons")
def seasons():
    import pyarrow.parquet as pq
    p = Path("data/shots.parquet")
    if not p.exists():
        raise HTTPException(503, "Data not ready")
    # Read only the season column — minimal memory
    table = pq.read_table(p, columns=["season"])
    return sorted(table.column("season").unique().to_pylist(), reverse=True)


@app.get("/api/teams")
def teams(
    league: str = Query(None),
    season: str = Query(None),
):
    if season:
        df = _shots_query(league=league, season=season,
                          columns=["match_id", "team", "league", "season",
                                   "goal", "xg", "opponent", "shots"])
        result = D.build_team_table(df)
        del df
        gc.collect()
        if league:
            result = result[result["league"] == league]
        return result.fillna(0).to_dict(orient="records")

    df = _df("teams")
    if league:
        df = df[df["league"] == league]
    return df.fillna(0).to_dict(orient="records")


@app.get("/api/players")
def players(
    league: str = Query(None),
    team:   str = Query(None),
    season: str = Query(None),
    min_shots: int = Query(5),
    limit: int = Query(50),
):
    if season:
        cols = ["player", "team", "league", "season", "goal", "xg"]
        df = _shots_query(league=league, team=team, season=season, columns=cols, limit=999999)
        agg = (df.groupby(["player", "team", "league", "season"])
                 .agg(shots=("xg", "count"), goals=("goal", "sum"),
                      xg=("xg", "sum"), xg_per_shot=("xg", "mean"))
                 .reset_index())
        del df
        gc.collect()
    else:
        agg = _df("players").copy()
        if league: agg = agg[agg["league"] == league]
        if team: agg = agg[agg["team"] == team]

    agg["xg_diff"] = (agg["goals"] - agg["xg"]).round(2)
    agg["xg"] = agg["xg"].round(2)
    agg["xg_per_shot"] = agg["xg_per_shot"].round(3)
    agg = agg.sort_values("xg", ascending=False)
    agg = agg[agg["shots"] >= min_shots].head(limit)
    return agg.fillna(0).to_dict(orient="records")


@app.get("/api/watchlist")
def watchlist(
    league: str = Query(None),
    season: str = Query(None),
    limit:  int = Query(50),
):
    df = _df("watchlist")
    if league: df = df[df["league"] == league]
    if season: df = df[df["season"] == season]
    return df.head(limit).fillna(0).to_dict(orient="records")


@app.get("/api/watchlist/form")
def watchlist_form(
    league: str = Query(None),
    limit:  int = Query(50),
):
    """
    Serve the pre-built 'Players to Watch' table.
    """
    df = _df("watchlist_form")
    if league:
        df = df[df["league"] == league]
    return df.head(limit).fillna(0).to_dict(orient="records")


@app.get("/api/matches")
def matches(
    league: str = Query(None),
    team:   str = Query(None),
    season: str = Query(None),
    limit:  int = Query(30),
):
    df = _df("matches")
    if league: df = df[df["league"] == league]
    if season: df = df[df["season"] == season]
    if team:
        df = df[(df["home_team"] == team) | (df["away_team"] == team)]
    return df.head(limit).fillna(0).to_dict(orient="records")


@app.get("/api/timeline/{match_id:path}")
def timeline(match_id: str):
    df = _df("timeline")
    sub = df[df["match_id"].astype(str) == match_id]
    if sub.empty:
        raise HTTPException(404, "Match not found")
    return sub.fillna(0).to_dict(orient="records")


@app.get("/api/shots")
def shots(
    league: str = Query(None),
    team: str = Query(None),
    player: str = Query(None),
    season: str = Query(None),
    match_id: str = Query(None),
    limit: int = Query(2000),
):

    df = _shots_query(
        league=league, team=team, player=player,
        season=season, match_id=match_id, limit=limit,
    )
    records = df.fillna(0).to_dict(orient="records")
    del df
    gc.collect()
    return records


@app.get("/api/refresh")
def refresh(background_tasks: BackgroundTasks):
    def _do():
        D.refresh([D.CURRENT_SEASON])
        _invalidate_cache()

    background_tasks.add_task(_do)
    return {"status": "refresh started"}

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/{full_path:path}")
def spa(full_path: str):
    return FileResponse("static/index.html")