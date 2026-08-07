"""
api.py - FastAPI backend
"""

from pathlib import Path
from datetime import datetime
import threading

import duckdb
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from startup import ensure_data

threading.Thread(target=ensure_data, daemon=True).start()

con = duckdb.connect()
con.execute("SET memory_limit='300MB'")
con.execute("SET threads=2")

def _q(sql: str, params: list = None):
    """Execute a query and return rows as a list of dicts."""
    p = Path("data")
    if not any(p.glob("*.parquet")):
        raise HTTPException(503, "Data not ready — run: python data.py")
    try:
        result = con.execute(sql, params or []).fetchdf()
        return result.fillna(0).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(500, str(e))

def _last_updated():
    p = Path("data/last_updated.txt")
    return p.read_text().strip() if p.exists() else "never"

def _parquet(name: str) -> str:
    """Return the parquet path as a quoted string for use in SQL."""
    p = Path(f"data/{name}.parquet")
    if not p.exists():
        raise HTTPException(503, f"Data not ready — missing: {name}.parquet")
    return f"'data/{name}.parquet'"

app = FastAPI(title="xG Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/status")
def status():
    return {"last_updated": _last_updated(), "status": "ok"}


@app.get("/api/leagues")
def leagues():
    import data as D
    return list(D.LEAGUES.values())


@app.get("/api/seasons")
def seasons():
    p = _parquet("shots")
    rows = con.execute(f"SELECT DISTINCT season FROM {p} ORDER BY season DESC").fetchall()
    return [r[0] for r in rows]


@app.get("/api/teams")
def teams(
    league: str = Query(None),
    season: str = Query(None),
):
    if season:
        p = _parquet("shots")
        filters, vals = [], []
        if league: filters.append("league = ?"); vals.append(league)
        filters.append("season = ?"); vals.append(season)
        where = "WHERE " + " AND ".join(filters)

        sql = f"""
            WITH base AS (
                SELECT team, league, match_id, goal, xg, opponent
                FROM {p} {where}
            ),
            atk AS (
                SELECT team, league,
                    COUNT(DISTINCT match_id) AS matches,
                    COUNT(*) AS shots,
                    SUM(goal) AS goals,
                    SUM(xg) AS xg
                FROM base
                GROUP BY team, league
            ),
            def_ AS (
                SELECT opponent AS team,
                    SUM(goal) AS goals_against,
                    SUM(xg) AS xga
                FROM base
                GROUP BY opponent
            )
            SELECT a.team, a.league, a.matches, a.shots, a.goals,
                ROUND(a.xg, 2) AS xg,
                ROUND(a.goals - a.xg, 2) AS xg_diff,
                ROUND(d.xga, 2) AS xga,
                ROUND(d.goals_against - d.xga, 2) AS xga_diff,
                ROUND(a.xg / NULLIF(a.shots, 0), 3) AS xg_per_shot
            FROM atk a
            LEFT JOIN def_ d ON a.team = d.team
            ORDER BY xg DESC
        """
        return _q(sql, vals)

    p = _parquet("teams")
    filters, vals = [], []
    if league: filters.append("league = ?"); vals.append(league)
    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    return _q(f"SELECT * FROM {p} {where} ORDER BY xg DESC", vals)


@app.get("/api/players")
def players(
    league:    str = Query(None),
    team:      str = Query(None),
    season:    str = Query(None),
    min_shots: int = Query(5),
    limit:     int = Query(50),
):
    limit = min(limit, 200)

    if season:
        p = _parquet("shots")
        filters, vals = ["TRUE"], []
        if league: filters.append("league = ?"); vals.append(league)
        if team:   filters.append("team = ?");   vals.append(team)
        filters.append("season = ?"); vals.append(season)
        where = "WHERE " + " AND ".join(filters)
        sql = f"""
            SELECT player, team, league,
                COUNT(*) AS shots,
                SUM(goal) AS goals,
                ROUND(SUM(xg), 2) AS xg,
                ROUND(SUM(goal) - SUM(xg), 2) AS xg_diff,
                ROUND(AVG(xg), 3) AS xg_per_shot
            FROM {p} {where}
            GROUP BY player, team, league
            HAVING COUNT(*) >= {min_shots}
            ORDER BY xg DESC
            LIMIT {limit}
        """
        return _q(sql, vals)

    p = _parquet("players")
    filters, vals = [], []
    if league: filters.append("league = ?"); vals.append(league)
    if team:   filters.append("team = ?");   vals.append(team)
    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    sql = f"""
        SELECT * FROM {p} {where}
        WHERE shots >= {min_shots}
        ORDER BY xg DESC
        LIMIT {limit}
    """
    # Note: can't have two WHERE clauses, rebuild cleanly
    conds = [f"shots >= {min_shots}"]
    if league: conds.append("league = ?")
    if team:   conds.append("team = ?")
    vals = []
    if league: vals.append(league)
    if team:   vals.append(team)
    where2 = "WHERE " + " AND ".join(conds)
    return _q(f"SELECT * FROM {p} {where2} ORDER BY xg DESC LIMIT {limit}", vals)


@app.get("/api/watchlist")
def watchlist(
    league: str = Query(None),
    season: str = Query(None),
    limit:  int = Query(50),
):
    p = _parquet("watchlist")
    conds, vals = [], []
    if league: conds.append("league = ?"); vals.append(league)
    if season: conds.append("season = ?"); vals.append(season)
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    return _q(f"SELECT * FROM {p} {where} ORDER BY rating DESC LIMIT {limit}", vals)


@app.get("/api/watchlist/form")
def watchlist_form(
    league: str = Query(None),
    limit:  int = Query(50),
):
    p = _parquet("watchlist_form")
    conds, vals = [], []
    if league: conds.append("league = ?"); vals.append(league)
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    return _q(f"SELECT * FROM {p} {where} ORDER BY rating DESC LIMIT {limit}", vals)


@app.get("/api/matches")
def matches(
    league: str = Query(None),
    team:   str = Query(None),
    season: str = Query(None),
    limit:  int = Query(30),
):
    p = _parquet("matches")
    conds, vals = [], []
    if league: conds.append("league = ?");                            vals.append(league)
    if season: conds.append("season = ?");                            vals.append(season)
    if team:   conds.append("(home_team = ? OR away_team = ?)");      vals.extend([team, team])
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    return _q(f"SELECT * FROM {p} {where} ORDER BY date DESC LIMIT {limit}", vals)


@app.get("/api/timeline/{match_id:path}")
def timeline(match_id: str):
    p = _parquet("timeline")
    rows = _q(
        f"SELECT * FROM {p} WHERE CAST(match_id AS VARCHAR) = ?",
        [match_id]
    )
    if not rows:
        raise HTTPException(404, "Match not found")
    return rows


@app.get("/api/shots")
def shots(
    league:   str = Query(None),
    team:     str = Query(None),
    player:   str = Query(None),
    season:   str = Query(None),
    match_id: str = Query(None),
    limit:    int = Query(2000),
):
    limit = min(limit, 3000)
    p = _parquet("shots")

    cols = """match_id, match_date, player, team, opponent, league, season,
              minute, x, y, goal, situation, shot_type, xg, understat_xg"""

    conds, vals = [], []
    if league:   conds.append("league = ?");                         vals.append(league)
    if season:   conds.append("season = ?");                         vals.append(season)
    if team:     conds.append("team = ?");                           vals.append(team)
    if player:   conds.append("player = ?");                         vals.append(player)
    if match_id: conds.append("CAST(match_id AS VARCHAR) = ?");      vals.append(match_id)

    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    return _q(f"SELECT {cols} FROM {p} {where} LIMIT {limit}", vals)


@app.get("/api/refresh")
def refresh():
    return {
        "status": "Refresh is handled by the daily GitHub Actions cron job. "
                  "To trigger manually, go to your repo → Actions → Daily Data Refresh → Run workflow."
    }


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/{full_path:path}")
def spa(full_path: str):
    p = Path(f"static/{full_path}")
    if p.exists() and p.is_file():
        return FileResponse(str(p))
    return FileResponse("static/index.html")