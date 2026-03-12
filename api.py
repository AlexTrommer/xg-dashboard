"""
api.py — FastAPI backend
Run locally:  uvicorn api:app --reload --port 8000
"""
from pathlib import Path
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from apscheduler.schedulers.background import BackgroundScheduler

import data as D
import threading
from startup import ensure_data

threading.Thread(target=ensure_data, daemon=True).start()

# ── In-memory cache ────────────────────────────────────────────────────────────
_CACHE: dict = {}

def _df(name: str) -> pd.DataFrame:
    if name not in _CACHE:
        p = Path(f"data/{name}.parquet")
        if not p.exists():
            raise HTTPException(503, "Data not ready — run: python data.py")
        _CACHE[name] = pd.read_parquet(p)
    return _CACHE[name]

def _invalidate_cache():
    _CACHE.clear()
app = FastAPI(title="xG Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Refresh current season every 6 hours
scheduler = BackgroundScheduler()
scheduler.add_job(lambda: D.refresh([D.CURRENT_SEASON]), "interval",
                  hours=24, id="auto_refresh")
scheduler.start()


def _df(name: str) -> pd.DataFrame:
    p = Path(f"data/{name}.parquet")
    if not p.exists():
        raise HTTPException(503, "Data not ready — run: python data.py")
    return pd.read_parquet(p)


def _last_updated():
    p = Path("data/last_updated.txt")
    return p.read_text().strip() if p.exists() else "never"


# ── Endpoints ──────────────────────────────────────────────────────────────────

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
    table = pq.read_table(p, columns=["season"])
    return sorted(table.column("season").unique().to_pylist(), reverse=True)


@app.get("/api/teams")
def teams(league: str = Query(None), season: str = Query(None)):
    df = _df("teams")
    if league:  df = df[df["league"] == league]
    if season:
        shots = _df("shots")
        if league:  shots = shots[shots["league"] == league]
        shots = shots[shots["season"] == season]
        df = D.build_team_table(shots)
        if league:  df = df[df["league"] == league]
    return df.fillna(0).to_dict(orient="records")


@app.get("/api/players")
def players(
    league: str = Query(None),
    team:   str = Query(None),
    season: str = Query(None),
    min_shots: int = Query(5),
    limit: int = Query(50),
):
    shots = _df("shots")
    if league: shots = shots[shots["league"] == league]
    if team:   shots = shots[shots["team"]   == team]

    if season:
        shots = shots[shots["season"] == season]
        df = (shots.groupby(["player", "team", "league", "season"])
                   .agg(shots=("xg","count"), goals=("goal","sum"),
                        xg=("xg","sum"), xg_per_shot=("xg","mean"))
                   .reset_index())
    else:
        df = (shots.groupby(["player", "team", "league"])
                   .agg(shots=("xg","count"), goals=("goal","sum"),
                        xg=("xg","sum"), xg_per_shot=("xg","mean"))
                   .reset_index())
        df["season"] = None

    df["xg_diff"]     = (df["goals"] - df["xg"]).round(2)
    df["xg"]          = df["xg"].round(2)
    df["xg_per_shot"] = df["xg_per_shot"].round(3)
    df = df.sort_values("xg", ascending=False).reset_index(drop=True)
    df = df[df["shots"] >= min_shots].head(limit)
    return df.fillna(0).to_dict(orient="records")


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
    Players to Watch — last 5 games vs prior 5 games.

    Rating is based on outcome-independent metrics only:
      npxG/90  (60%) — non-penalty xG per 90 mins, estimated from shots
      xG/shot  (40%) — shot quality / positioning

    Minutes are estimated as shots * avg_mins_per_shot (calibrated to ~70 mins
    per 3.5 shots for a typical starting forward). This avoids needing lineup data.

    Only players who appeared in at least 3 of their last 5 games are shown,
    and only if they had at least 3 shots in the window (noise filter).
    Goals are shown for context but do NOT affect the rating.
    """
    import numpy as np

    shots = _df("shots").copy()
    shots["match_date"] = pd.to_datetime(shots["match_date"], errors="coerce")

    if league:
        shots = shots[shots["league"] == league]

    # Current season only — don't mix seasons for "form"
    current_season = shots["season"].max()
    shots = shots[shots["season"] == current_season]

    if shots.empty:
        return []

    # ── Per-player game windows ───────────────────────────────────────────────
    # For each player, rank their matches by date and split last-5 / prior-5.
    # We tag each shot with the player's match rank (most recent = rank 1).
    shots = shots.sort_values("match_date")

    # Assign a per-player match rank (1 = most recent)
    shots["_match_rank"] = (
        shots.groupby(["player", "match_id"])["match_date"]
             .transform("first")
             .groupby(shots["player"])
             .rank(method="dense", ascending=False)
             .astype(int)
    )

    recent = shots[shots["_match_rank"] <= 5]
    prior  = shots[(shots["_match_rank"] > 5) & (shots["_match_rank"] <= 10)]

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def aggregate(df):
        grp = (df.groupby(["player", "team", "league"])
                 .agg(
                     shots        = ("xg",      "count"),
                     goals        = ("goal",     "sum"),
                     xg           = ("xg",       "sum"),
                     xg_per_shot  = ("xg",       "mean"),
                     games        = ("match_id", "nunique"),
                     np_xg        = ("xg",       lambda x:
                                     x[df.loc[x.index, "situation"] != "Penalty"].sum()),
                 )
                 .reset_index())
        grp["xg"]          = grp["xg"].round(2)
        grp["np_xg"]       = grp["np_xg"].round(2)
        grp["xg_per_shot"] = grp["xg_per_shot"].round(3)
        grp["xg_diff"]     = (grp["goals"] - grp["xg"]).round(2)
        # Estimate minutes: calibrated so 3.5 shots ≈ 70 min (avg starting forward)
        grp["est_mins"]    = (grp["shots"] * 20.0).clip(upper=grp["games"] * 96)
        grp["np_xg_p90"]   = ((grp["np_xg"] / grp["est_mins"].clip(lower=1)) * 90).round(3)
        return grp

    r = aggregate(recent)
    p = aggregate(prior)

    # ── Filter: min 3 games, min 3 shots, min ~150 estimated minutes ────────────
    # 150 mins ≈ 2 full games worth of estimated playing time.
    # This disqualifies impact subs who played 3 x 20 mins and got 1 shot each.
    r = r[(r["games"] >= 3) & (r["shots"] >= 3) & (r["est_mins"] >= 150)].copy()
    if r.empty:
        return []

    # ── Rating: npxG/90 (60%) + xG/shot (40%) — no outcome dependency ─────────
    def norm(col):
        mn, mx = col.min(), col.max()
        if mx == mn: return pd.Series(0.5, index=col.index)
        return (col - mn) / (mx - mn)

    r["_qual"] = r["xg_per_shot"].clip(upper=0.5)
    r["_rate"] = r["np_xg_p90"].clip(upper=1.0)
    raw = norm(r["_rate"]) * 0.60 + norm(r["_qual"]) * 0.40
    r["rating"] = (norm(raw) * 100).round(1)
    r = r.drop(columns=["_qual", "_rate"])
    r = r.sort_values("rating", ascending=False).head(limit).reset_index(drop=True)

    # ── Merge prior window ────────────────────────────────────────────────────
    prev_cols = ["shots", "goals", "xg", "xg_per_shot", "xg_diff", "np_xg_p90", "games"]
    p_ren = p.rename(columns={c: f"prev_{c}" for c in prev_cols})

    # Prior rating
    if len(p_ren) > 1:
        p2 = p.copy()
        p2["_qual"] = p2["xg_per_shot"].clip(upper=0.5)
        p2["_rate"] = p2["np_xg_p90"].clip(upper=1.0)
        raw2 = norm(p2["_rate"]) * 0.60 + norm(p2["_qual"]) * 0.40
        p2["prev_rating"] = (norm(raw2) * 100).round(1)
        p_ren = p_ren.merge(p2[["player", "team", "prev_rating"]],
                            on=["player", "team"], how="left")
    else:
        p_ren["prev_rating"] = None

    keep = ["player", "team"] + [c for c in p_ren.columns
                                  if c.startswith("prev_")]
    merged = r.merge(p_ren[keep], on=["player", "team"], how="left")
    merged["rating_delta"] = (
        merged["rating"] - merged["prev_rating"].fillna(merged["rating"])
    ).round(1)
    merged["window_games"] = 5
    merged["season"]       = current_season

    return merged.fillna(0).to_dict(orient="records")


@app.get("/api/matches")
def matches(
    league: str = Query(None),
    team:   str = Query(None),
    season: str = Query(None),
    limit:  int = Query(30),
):
    df = _df("matches")
    if league:  df = df[df["league"] == league]
    if season:  df = df[df["season"] == season]
    if team:    df = df[(df["home_team"] == team) | (df["away_team"] == team)]
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
    league:   str = Query(None),
    team:     str = Query(None),
    player:   str = Query(None),
    season:   str = Query(None),
    match_id: str = Query(None),
    limit:    int = Query(2000),
):
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    p = Path("data/shots.parquet")
    if not p.exists():
        raise HTTPException(503, "Data not ready")

    cols = ["match_id","match_date","player","team","opponent","league","season",
            "minute","x","y","goal","situation","shot_type","xg","understat_xg"]

    # Read only needed columns
    available = pq.read_schema(p).names
    read_cols = [c for c in cols if c in available]

    # Build pyarrow filters for partition pruning
    filters = []
    if league:   filters.append(("league",  "=", league))
    if season:   filters.append(("season",  "=", season))
    if team:     filters.append(("team",    "=", team))
    if player:   filters.append(("player",  "=", player))

    df = pq.read_table(p, columns=read_cols,
                       filters=filters if filters else None).to_pandas()

    if match_id:
        df = df[df["match_id"].astype(str) == match_id]

    for c in cols:
        if c not in df.columns:
            df[c] = ""

    return df.head(limit).fillna(0).to_dict(orient="records")

@app.get("/api/refresh")
def refresh():
    _invalidate_cache()
    threading.Thread(target=D.refresh, daemon=True).start()
    return {"status": "refresh started"}

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/{full_path:path}")
def spa(full_path: str):
    return FileResponse("static/index.html")