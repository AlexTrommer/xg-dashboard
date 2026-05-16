"""
data.py — Understat top-5 league data puller + XGBoost xG model
"""
import asyncio
import gc
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
 
warnings.filterwarnings("ignore")
Path("data").mkdir(exist_ok=True)
 
LEAGUES = {
    "EPL":        "Premier League",
    "La_liga":    "La Liga",
    "Bundesliga": "Bundesliga",
    "Serie_A":    "Serie A",
    "Ligue_1":    "Ligue 1",
}
CURRENT_SEASON = "2025"
GOAL_X         = 100.0
GOAL_Y_CENTER  = 50.0
GOAL_WIDTH_PCT = 11.6
PENALTY_XG     = 0.76
 
FEATURES = [
    # geometry
    "distance", "angle_deg", "distance_sq", "distance_x_angle",
    "dist_x_angle_sq",
    # shot type
    "is_header", "is_set_piece", "is_rebound",
    "is_cross", "is_throughball",
    # interactions: shot type x position
    "header_x_dist", "header_x_angle",
    "cross_x_dist", "throughball_x_dist",
    "rebound_x_dist",
    # zone and context
    "shot_zone", "is_big_chance", "minute_norm",
    # y-axis centrality
    "central_threat",
    # proxy features
    "fast_break", "assisted_header", "is_cutback",
    "is_penalty_area", "weak_angle_header",
]
 
 
# ── Fetch ──────────────────────────────────────────────────────────────────────
async def _fetch_league(league: str, season: str) -> list:
    try:
        import aiohttp
        import understat
        all_shots = []
        async with aiohttp.ClientSession() as session:
            u = understat.Understat(session)
            results = await u.get_league_results(league, season)
            for match in results:
                mid = match.get("id")
                if not mid:
                    continue
                try:
                    shots = await u.get_match_shots(mid)
                    for side in ("h", "a"):
                        for s in shots.get(side, []):
                            s["match_id"] = mid
                            s["h_team"]   = match.get("h", {}).get("title", "")
                            s["a_team"]   = match.get("a", {}).get("title", "")
                            s["date"]     = match.get("datetime", "")
                            s["h_a"]      = side
                            all_shots.append(s)
                except Exception:
                    continue
        return all_shots
    except Exception as e:
        print(f"  [{league} {season}] error: {e}")
        return []
 
 
def pull_shots(seasons=None) -> pd.DataFrame:
    if seasons is None:
        seasons = [str(y) for y in range(2020, int(CURRENT_SEASON) + 1)]
    rows = []
    for league_key, league_name in LEAGUES.items():
        for season in seasons:
            print(f"  {league_name} {season}…", end=" ", flush=True)
            shots = asyncio.run(_fetch_league(league_key, season))
            print(f"{len(shots)} shots")
            for s in shots:
                h_a = s.get("h_a", "h")
                rows.append({
                    "id"          : f"{league_key}_{s.get('id')}",
                    "season"      : season,
                    "league"      : league_name,
                    "match_id"    : f"{league_key}_{s.get('match_id')}",
                    "player"      : s.get("player"),
                    "player_id"   : s.get("player_id"),
                    "team"        : s.get("h_team") if h_a == "h" else s.get("a_team"),
                    "opponent"    : s.get("a_team") if h_a == "h" else s.get("h_team"),
                    "h_a"         : h_a,
                    "minute"      : int(s.get("minute", 0) or 0),
                    "x"           : float(s.get("X", 0) or 0) * 100,
                    "y"           : float(s.get("Y", 0) or 0) * 100,
                    "goal"        : int(s.get("result", "") == "Goal"),
                    "situation"   : s.get("situation", "OpenPlay"),
                    "shot_type"   : s.get("shotType", ""),
                    "last_action" : s.get("lastAction", ""),
                    "understat_xg": float(s.get("xG", 0) or 0),
                    "match_date"  : (s.get("date", "") or "")[:10],
                })
            time.sleep(1.5)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No shots fetched — check your internet connection and Understat availability")
    df = df.drop_duplicates(subset=["id"])
    print(f"\nTotal: {len(df):,} shots | {df['goal'].sum():,} goals")
    return df
 
 
# ── Features ───────────────────────────────────────────────────────────────────
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    n_before = len(df)
    is_pen   = df["situation"] == "Penalty"
    valid_x  = (df["x"] >= 17) & (df["x"] <= 101)
    valid_y  = (df["y"] >= 2)  & (df["y"] <= 98)
    df = df[is_pen | (valid_x & valid_y)].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  [filter] dropped {n_dropped:,} out-of-range shots "
              f"({n_dropped/n_before*100:.2f}% of {n_before:,})")
 
    # Geometry
    dx = GOAL_X - df["x"]
    dy = df["y"] - GOAL_Y_CENTER
    df["distance"]         = np.sqrt(dx**2 + dy**2)
    half = GOAL_WIDTH_PCT / 2
    num  = GOAL_WIDTH_PCT * dx
    den  = np.where(dx**2 + dy**2 - half**2 <= 0, 1e-6, dx**2 + dy**2 - half**2)
    df["angle_deg"]        = np.degrees(np.arctan(num / den)).clip(lower=0)
    df["distance_sq"]      = df["distance"] ** 2
    df["distance_x_angle"] = df["distance"] * df["angle_deg"]
    df["dist_x_angle_sq"]  = df["distance"] * df["angle_deg"] ** 2
 
    sit   = df["situation"].str.lower().fillna("")
    stype = df["shot_type"].str.lower().fillna("")
    la    = df["last_action"].str.lower().fillna("") if "last_action" in df.columns \
            else pd.Series("", index=df.index)
 
    df["is_header"]      = (stype == "head").astype(int)
    df["is_counter"]     = pd.Series(0, index=df.index)
    df["is_set_piece"]   = sit.str.contains("freekick|corner|setpiece").astype(int)
    df["is_penalty"]     = sit.str.contains("penalty").astype(int)
    df["is_cross"]       = la.str.contains("cross").astype(int)
    df["is_throughball"] = la.str.contains("throughball|through ball").astype(int)
    df["is_rebound"]     = la.str.contains("rebound|saves|blockedshot|rebound").astype(int)
 
    df["header_x_dist"]      = df["is_header"]      * df["distance"]
    df["header_x_angle"]     = df["is_header"]      * df["angle_deg"]
    df["cross_x_dist"]       = df["is_cross"]       * df["distance"]
    df["throughball_x_dist"] = df["is_throughball"] * df["distance"]
    df["rebound_x_dist"]     = df["is_rebound"]     * df["distance"]
 
    df["is_big_chance"]  = ((df["distance"] < 12) & (df["angle_deg"] > 20)).astype(int)
    df["central_threat"] = 1.0 - (df["y"] - 50.0).abs() / 50.0
 
    def zone(r):
        if r["x"] >= 95: return 0
        if r["x"] >= 83 and 30 <= r["y"] <= 70: return 1
        return 2
    df["shot_zone"]   = df.apply(zone, axis=1)
    df["minute_norm"] = (df["minute"].fillna(45) / 120).clip(0, 1)
 
    df["is_penalty_area"] = ((df["x"] >= 83) & (df["y"] >= 22) & (df["y"] <= 78)).astype(int)
 
    transition_actions = {"TakeOn", "BallRecovery", "Throughball", "throughball"}
    df["fast_break"] = (
        (df["situation"] == "OpenPlay") &
        (la.isin(transition_actions) | la.str.contains("throughball", case=False, na=False))
    ).astype(int)
    df["assisted_header"]  = (df["is_header"] & df["is_cross"]).astype(int)
    df["is_cutback"]       = (
        (df["is_cross"] == 1) & (df["x"] > 88) &
        ((df["y"] < 28) | (df["y"] > 72))
    ).astype(int)
    df["weak_angle_header"] = (df["is_header"] & (df["angle_deg"] < 10)).astype(int)
 
    return df
 
 
# ── Model ──────────────────────────────────────────────────────────────────────
def _route_predict(df, models, router, feature_map=None):
    if feature_map is None:
        feature_map = {}
    preds = np.full(len(df), np.nan)
    df = df.reset_index(drop=True)
    df["_model_key"] = df["situation"].map(router).fillna("global")
    for key, grp in df.groupby("_model_key"):
        m    = models.get(key, models["global"])
        feat = feature_map.get(key, feature_map.get("global", FEATURES))
        valid = grp.dropna(subset=feat)
        if len(valid):
            preds[valid.index] = m.predict_proba(valid[feat].values)[:, 1]
    return preds
 
 
def train(df: pd.DataFrame):
    df_tr = df[df["is_penalty"] == 0].dropna(subset=FEATURES + ["goal"])
    X, y  = df_tr[FEATURES].values, df_tr["goal"].values
    neg, pos = (y == 0).sum(), (y == 1).sum()
    spw = round(neg / pos, 2)
 
    params_path = Path("data/best_params.json")
    if params_path.exists():
        import json
        best = json.loads(params_path.read_text())
        print(f"  Loaded tuned params from {params_path}")
    else:
        best = dict(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.75, colsample_bytree=0.75,
            min_child_weight=10,
        )
        print("  Using default params (run train.py to tune)")
 
    xgb_kwargs = dict(
        gamma=0.2, reg_alpha=0.2, reg_lambda=1.5,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, n_jobs=2, tree_method="hist",
    )
 
    model_path = Path("model.pkl")
    if model_path.exists():
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict) and "models" in bundle:
            print("  Loaded per-situation model bundle from model.pkl")
            models      = bundle["models"]
            router      = bundle["router"]
            feature_map = bundle.get("feature_map", {})
            valid       = df.dropna(subset=FEATURES)
            preds       = _route_predict(valid, models, router, feature_map)
            df.loc[valid.index, "xg"] = preds
            del valid, preds
            gc.collect()
            df.loc[df["is_penalty"] == 1, "xg"] = PENALTY_XG
            df["xg"] = df["xg"].fillna(df["understat_xg"])
            return bundle, df
        else:
            print("  Old model format — retraining…")
 
    print("  Training global model (run train.py for per-situation models)…")
    base  = XGBClassifier(**best, **xgb_kwargs)
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    model.fit(X, y)
    del X, y, df_tr
    gc.collect()
 
    models = {"global": model}
    router = {}
    bundle = {"models": models, "router": router}
 
    valid = df.dropna(subset=FEATURES)
    preds = _route_predict(valid, models, router, {})
    df.loc[valid.index, "xg"] = preds
    del valid, preds
    gc.collect()
    df.loc[df["is_penalty"] == 1, "xg"] = PENALTY_XG
    df["xg"] = df["xg"].fillna(df["understat_xg"])
    return bundle, df
 
 
# ── Aggregations ───────────────────────────────────────────────────────────────
def possession_xg(vals):
    p = 1.0
    for v in vals:
        p *= (1.0 - float(v))
    return 1.0 - p
 
 
def build_team_table(df):
    poss = (df.groupby(["match_id", "team", "league"])
              .agg(shots=("xg","count"), goals=("goal","sum"),
                   xg_add=("xg","sum"), xg_poss=("xg", possession_xg))
              .reset_index())
    t = (poss.groupby(["team", "league"])
             .agg(matches     = ("match_id","nunique"),
                  shots       = ("shots","sum"),
                  goals       = ("goals","sum"),
                  xg          = ("xg_add","sum"),
                  xg_poss     = ("xg_poss","sum"),
                  possessions = ("match_id","count"))
             .reset_index())
 
    df_opp = df.copy().rename(columns={"team": "_opp", "opponent": "team"})
    poss_a = (df_opp.groupby(["match_id", "team", "league"])
                    .agg(goals_against=("goal","sum"),
                         xga          =("xg","sum"),
                         xga_poss     =("xg", possession_xg))
                    .reset_index())
    ta = (poss_a.groupby(["team", "league"])
               .agg(goals_against=("goals_against","sum"),
                    xga          =("xga","sum"),
                    xga_poss     =("xga_poss","sum"))
               .reset_index())
    del df_opp, poss_a
    gc.collect()
 
    t = t.merge(ta, on=["team","league"], how="left")
    del ta
    gc.collect()
 
    t["xg_diff"]      = (t["goals"]         - t["xg"]).round(2)
    t["xga_diff"]     = (t["goals_against"] - t["xga"]).round(2)
    t["xg"]           = t["xg"].round(2)
    t["xg_poss"]      = t["xg_poss"].round(2)
    t["xga"]          = t["xga"].round(2)
    t["xga_poss"]     = t["xga_poss"].round(2)
    t["xg_per_poss"]  = (t["xg_poss"]  / t["possessions"]).round(3)
    t["xga_per_poss"] = (t["xga_poss"] / t["possessions"]).round(3)
    t["xg_per_shot"]  = (t["xg"]       / t["shots"]).round(3)
    return t.sort_values("xg", ascending=False).reset_index(drop=True)
 
 
def build_player_table(df):
    p = (df.groupby(["player", "team", "league"])
           .agg(shots=("xg","count"), goals=("goal","sum"),
                xg=("xg","sum"), xg_per_shot=("xg","mean"))
           .reset_index())
    p["xg_diff"]     = (p["goals"] - p["xg"]).round(2)
    p["xg"]          = p["xg"].round(2)
    p["xg_per_shot"] = p["xg_per_shot"].round(3)
    return p.sort_values("xg", ascending=False).reset_index(drop=True)
 
 
def build_watchlist(df, min_goals=5):
    p = (df.groupby(["player", "team", "league", "season"])
           .agg(shots      = ("xg",   "count"),
                goals      = ("goal", "sum"),
                xg         = ("xg",   "sum"),
                xg_per_shot= ("xg",   "mean"))
           .reset_index())
    p = p[p["goals"] >= min_goals].copy()
    if p.empty:
        return p
 
    p["xg"]          = p["xg"].clip(lower=0.01)
    p["xg_per_shot"] = p["xg_per_shot"].round(3)
    p["xg_diff"]     = (p["goals"] - p["xg"]).round(2)
 
    p["efficiency"] = (p["goals"] / p["xg"]).clip(upper=2.0)
    p["quality"]    = p["xg_per_shot"].clip(upper=0.5)
    p["volume"]     = np.log1p(p["shots"]) / np.log1p(p["shots"].max())
 
    def norm(col):
        mn, mx = col.min(), col.max()
        if mx == mn:
            return pd.Series(0.5, index=col.index)
        return (col - mn) / (mx - mn)
 
    p["eff_n"] = norm(p["efficiency"])
    p["qlt_n"] = norm(p["quality"])
    p["vol_n"] = norm(p["volume"])
    p["rating"] = p["eff_n"] * 0.40 + p["qlt_n"] * 0.35 + p["vol_n"] * 0.25
    p["rating"] = (norm(p["rating"]) * 100).round(1)
    p = p.drop(columns=["efficiency", "quality", "volume", "eff_n", "qlt_n", "vol_n"])
    return p.sort_values("rating", ascending=False).reset_index(drop=True)
 
 
def build_watchlist_form(df) -> pd.DataFrame:
    """
    Pre-compute the 'Players to Watch' rolling form guide.
 
    Previously this ran on every API request, causing OOM on the free plan
    because it kept a full copy of shots in memory alongside all the
    intermediate groupby results. Now it runs once during refresh() and
    writes data/watchlist_form.parquet. The API just reads that file.
 
    Logic is unchanged from the original /api/watchlist/form endpoint.
    """
    shots = df.copy()
    shots["match_date"] = pd.to_datetime(shots["match_date"], errors="coerce")
 
    current_season = shots["season"].max()
    shots = shots[shots["season"] == current_season]
 
    if shots.empty:
        return pd.DataFrame()
 
    shots = shots.sort_values("match_date")
 
    # Per-player match rank (1 = most recent)
    shots["_match_rank"] = (
        shots.groupby(["player", "match_id"])["match_date"]
             .transform("first")
             .groupby(shots["player"])
             .rank(method="dense", ascending=False)
             .astype(int)
    )
 
    recent = shots[shots["_match_rank"] <= 5].copy()
    prior  = shots[(shots["_match_rank"] > 5) & (shots["_match_rank"] <= 10)].copy()
    del shots
    gc.collect()
 
    def aggregate(df_window):
        grp = (df_window.groupby(["player", "team", "league"])
                        .agg(
                            shots        = ("xg",      "count"),
                            goals        = ("goal",     "sum"),
                            xg           = ("xg",       "sum"),
                            xg_per_shot  = ("xg",       "mean"),
                            games        = ("match_id", "nunique"),
                            np_xg        = ("xg", lambda x:
                                            x[df_window.loc[x.index, "situation"] != "Penalty"].sum()),
                        )
                        .reset_index())
        grp["xg"]          = grp["xg"].round(2)
        grp["np_xg"]       = grp["np_xg"].round(2)
        grp["xg_per_shot"] = grp["xg_per_shot"].round(3)
        grp["xg_diff"]     = (grp["goals"] - grp["xg"]).round(2)
        grp["est_mins"]    = (grp["shots"] * 20.0).clip(upper=grp["games"] * 96)
        grp["np_xg_p90"]   = ((grp["np_xg"] / grp["est_mins"].clip(lower=1)) * 90).round(3)
        return grp
 
    r = aggregate(recent)
    p = aggregate(prior)
    del recent, prior
    gc.collect()
 
    # Filter: min 3 games, min 3 shots, min ~150 estimated minutes
    r = r[(r["games"] >= 3) & (r["shots"] >= 3) & (r["est_mins"] >= 150)].copy()
    if r.empty:
        return pd.DataFrame()
 
    def norm(col):
        mn, mx = col.min(), col.max()
        if mx == mn:
            return pd.Series(0.5, index=col.index)
        return (col - mn) / (mx - mn)
 
    r["_qual"] = r["xg_per_shot"].clip(upper=0.5)
    r["_rate"] = r["np_xg_p90"].clip(upper=1.0)
    raw = norm(r["_rate"]) * 0.60 + norm(r["_qual"]) * 0.40
    r["rating"] = (norm(raw) * 100).round(1)
    r = r.drop(columns=["_qual", "_rate"])
    r = r.sort_values("rating", ascending=False).reset_index(drop=True)
 
    # Merge prior window
    prev_cols = ["shots", "goals", "xg", "xg_per_shot", "xg_diff", "np_xg_p90", "games"]
    p_ren = p.rename(columns={c: f"prev_{c}" for c in prev_cols})
 
    if len(p) > 1:
        p2 = p.copy()
        p2["_qual"] = p2["xg_per_shot"].clip(upper=0.5)
        p2["_rate"] = p2["np_xg_p90"].clip(upper=1.0)
        raw2 = norm(p2["_rate"]) * 0.60 + norm(p2["_qual"]) * 0.40
        p2["prev_rating"] = (norm(raw2) * 100).round(1)
        p_ren = p_ren.merge(p2[["player", "team", "prev_rating"]],
                            on=["player", "team"], how="left")
        del p2
    else:
        p_ren["prev_rating"] = None
 
    del p
    gc.collect()
 
    keep = ["player", "team"] + [c for c in p_ren.columns if c.startswith("prev_")]
    merged = r.merge(p_ren[keep], on=["player", "team"], how="left")
    merged["rating_delta"] = (
        merged["rating"] - merged["prev_rating"].fillna(merged["rating"])
    ).round(1)
    merged["window_games"] = 5
    merged["season"]       = current_season
    return merged.fillna(0)
 
 
def build_match_timeline(df):
    rows = []
    for mid, grp in df.groupby("match_id"):
        teams = grp["team"].unique()
        if len(teams) < 2:
            continue
        date   = grp["match_date"].iloc[0]
        league = grp["league"].iloc[0]
        for team in teams:
            tdf = grp[grp["team"] == team].sort_values("minute")
            cum_xg, cum_g = 0.0, 0
            for _, row in tdf.iterrows():
                cum_xg += float(row["xg"]) if not pd.isna(row["xg"]) else 0
                cum_g  += int(row["goal"])
                rows.append({
                    "match_id" : mid,
                    "date"     : date,
                    "league"   : league,
                    "team"     : team,
                    "minute"   : row["minute"],
                    "cum_xg"   : round(cum_xg, 3),
                    "cum_goals": cum_g,
                    "xg"       : round(float(row["xg"]), 3) if not pd.isna(row["xg"]) else 0,
                    "goal"     : int(row["goal"]),
                    "player"   : row.get("player", ""),
                })
    return pd.DataFrame(rows)
 
 
def build_matches(df):
    rows = []
    for mid, grp in df.groupby("match_id"):
        teams = list(grp["team"].unique())
        if len(teams) < 2:
            continue
        t1, t2 = teams[0], teams[1]
        g1 = grp[grp["team"] == t1]
        g2 = grp[grp["team"] == t2]
        rows.append({
            "match_id"   : mid,
            "date"       : grp["match_date"].iloc[0],
            "league"     : grp["league"].iloc[0],
            "season"     : grp["season"].iloc[0],
            "home_team"  : t1,
            "away_team"  : t2,
            "home_goals" : int(g1["goal"].sum()),
            "away_goals" : int(g2["goal"].sum()),
            "home_xg"    : round(float(g1["xg"].fillna(0).sum()), 2),
            "away_xg"    : round(float(g2["xg"].fillna(0).sum()), 2),
        })
    return pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)
 
 
# ── Master refresh (memory-safe) ───────────────────────────────────────────────
def refresh(seasons=None):
    print(f"\n[{datetime.now():%H:%M:%S}] Refreshing…")
 
    # 1. Fetch + engineer (peak: raw shots in memory)
    df = engineer(pull_shots(seasons))
 
    # 2. Score with model (peak: df + model in memory simultaneously)
    bundle, df = train(df)
 
    # 3. Persist shots immediately so we can free engineering columns
    df.to_parquet("data/shots.parquet", index=False)
    print("  shots.parquet written ✓")
 
    # Save model
    with open("model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    del bundle
    gc.collect()
 
    # 4. Build and write each aggregate table, freeing intermediates as we go
    print("  Building team table…")
    build_team_table(df).to_parquet("data/teams.parquet", index=False)
    gc.collect()
 
    print("  Building player table…")
    build_player_table(df).to_parquet("data/players.parquet", index=False)
    gc.collect()
 
    print("  Building watchlist…")
    build_watchlist(df).to_parquet("data/watchlist.parquet", index=False)
    gc.collect()
 
    # NEW: pre-compute watchlist form so the API never has to run it live
    print("  Building watchlist form (Players to Watch)…")
    build_watchlist_form(df).to_parquet("data/watchlist_form.parquet", index=False)
    gc.collect()
 
    print("  Building match timeline…")
    build_match_timeline(df).to_parquet("data/timeline.parquet", index=False)
    gc.collect()
 
    print("  Building matches…")
    build_matches(df).to_parquet("data/matches.parquet", index=False)
 
    del df
    gc.collect()
 
    with open("data/last_updated.txt", "w") as f:
        f.write(datetime.now().isoformat())
    print("  Refresh complete ✓")
 
 
# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
 
    if "--rebuild-tables" in sys.argv:
        print("Rebuilding tables from existing shots.parquet…")
        df = pd.read_parquet("data/shots.parquet")
        _, df = train(df)
        build_team_table(df).to_parquet("data/teams.parquet", index=False)
        build_player_table(df).to_parquet("data/players.parquet", index=False)
        build_watchlist(df).to_parquet("data/watchlist.parquet", index=False)
        build_watchlist_form(df).to_parquet("data/watchlist_form.parquet", index=False)
        build_match_timeline(df).to_parquet("data/timeline.parquet", index=False)
        build_matches(df).to_parquet("data/matches.parquet", index=False)
        with open("data/last_updated.txt", "w") as f:
            f.write(datetime.now().isoformat())
        print("Done ✓")
 
    elif "--current-season-only" in sys.argv:
        print(f"Refreshing current season ({CURRENT_SEASON}) only…")
        new_df = engineer(pull_shots([CURRENT_SEASON]))
        old_df = pd.read_parquet("data/shots.parquet")
        old_df = old_df[old_df["season"] != int(CURRENT_SEASON)]
        df = pd.concat([old_df, new_df], ignore_index=True)
        df = df.drop_duplicates(subset=["id"])
        del old_df, new_df
        gc.collect()
        bundle, df = train(df)
        df.to_parquet("data/shots.parquet", index=False)
        with open("model.pkl", "wb") as f:
            pickle.dump(bundle, f)
        del bundle
        gc.collect()
        build_team_table(df).to_parquet("data/teams.parquet", index=False)
        build_player_table(df).to_parquet("data/players.parquet", index=False)
        build_watchlist(df).to_parquet("data/watchlist.parquet", index=False)
        build_watchlist_form(df).to_parquet("data/watchlist_form.parquet", index=False)
        build_match_timeline(df).to_parquet("data/timeline.parquet", index=False)
        build_matches(df).to_parquet("data/matches.parquet", index=False)
        with open("data/last_updated.txt", "w") as f:
            f.write(datetime.now().isoformat())
        print("Done ✓")
 
    else:
        refresh()