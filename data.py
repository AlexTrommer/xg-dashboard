"""
data.py — Understat top-5 league data puller + XGBoost xG model
"""
import asyncio, pickle, warnings, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

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

    # ── Coordinate sanity filter ───────────────────────────────────────────────
    # Understat coords: x=0-100 (100=goal being attacked), y=0-100 (50=centre).
    # x < 17  =  behind own penalty area edge — almost always noise or wrong team
    #            direction. Genuine half-pitch shots are ultra-rare and hurt the model.
    # x > 101 =  behind the goal line (rounding slack of 1).
    # y < 2 or y > 98  =  off the side of the pitch, pure coord errors.
    # Penalties are exempt (they're fixed at x~88 anyway, but just to be safe).
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

    # Shot type flags
    sit   = df["situation"].str.lower().fillna("")
    stype = df["shot_type"].str.lower().fillna("")
    la    = df["last_action"].str.lower().fillna("") if "last_action" in df.columns \
            else pd.Series("", index=df.index)

    df["is_header"]      = (stype == "head").astype(int)
    df["is_counter"]     = pd.Series(0, index=df.index)  # Understat doesn't tag counters
    df["is_set_piece"]   = sit.str.contains("freekick|corner|setpiece").astype(int)
    df["is_penalty"]     = sit.str.contains("penalty").astype(int)
    df["is_cross"]       = la.str.contains("cross").astype(int)
    df["is_throughball"] = la.str.contains("throughball|through ball").astype(int)
    df["is_rebound"]     = la.str.contains("rebound|saves|blockedshot|rebound").astype(int)

    # Interaction terms
    df["header_x_dist"]      = df["is_header"]      * df["distance"]
    df["header_x_angle"]     = df["is_header"]      * df["angle_deg"]
    df["cross_x_dist"]       = df["is_cross"]       * df["distance"]
    df["throughball_x_dist"] = df["is_throughball"] * df["distance"]
    df["rebound_x_dist"]     = df["is_rebound"]     * df["distance"]

    # Zone / context
    df["is_big_chance"]  = ((df["distance"] < 12) & (df["angle_deg"] > 20)).astype(int)
    df["central_threat"] = 1.0 - (df["y"] - 50.0).abs() / 50.0

    def zone(r):
        if r["x"] >= 95: return 0
        if r["x"] >= 83 and 30 <= r["y"] <= 70: return 1
        return 2
    df["shot_zone"]   = df.apply(zone, axis=1)
    df["minute_norm"] = (df["minute"].fillna(45) / 120).clip(0, 1)

    # Penalty area (finer than shot_zone)
    df["is_penalty_area"] = ((df["x"] >= 83) & (df["y"] >= 22) & (df["y"] <= 78)).astype(int)

    # Proxy features
    # fast_break: proxy for transition/counter using last_action context
    # TakeOn (dribble past defender), BallRecovery (won ball high up), Throughball
    # all indicate space behind the defence — the closest we can get without a counter tag
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
def _route_predict(df: pd.DataFrame, models: dict, router: dict,
                   feature_map: dict = None) -> np.ndarray:
    """Route each shot to its situation-specific model using per-situation features."""
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

    # Load best params from grid search if available, else use defaults
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

    # Check if a trained model bundle already exists from train.py
    model_path = Path("model.pkl")
    if model_path.exists():
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        # Support both old single-model and new bundle format
        if isinstance(bundle, dict) and "models" in bundle:
            print("  Loaded per-situation model bundle from data/model.pkl")
            models      = bundle["models"]
            router      = bundle["router"]
            feature_map = bundle.get("feature_map", {})
            valid       = df.dropna(subset=FEATURES)
            preds       = _route_predict(valid, models, router, feature_map)
            df.loc[valid.index, "xg"] = preds
            del valid, preds
            df.loc[df["is_penalty"] == 1, "xg"] = PENALTY_XG
            df["xg"] = df["xg"].fillna(df["understat_xg"])
            return bundle, df
        else:
            # Old single model — fall through to retrain
            print("  Old model format detected — retraining…")

    # No bundle found — train a simple global model
    print("  Training global model (run train.py for per-situation models)…")
    base  = XGBClassifier(**best, **xgb_kwargs)
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    model.fit(X, y)

    models = {"global": model}
    router = {}   # empty router — everything falls back to global
    bundle = {"models": models, "router": router}

    valid = df.dropna(subset=FEATURES)
    preds = _route_predict(valid, models, router, feature_map)
    df.loc[valid.index, "xg"] = preds
    df.loc[df["is_penalty"] == 1, "xg"] = PENALTY_XG
    df["xg"] = df["xg"].fillna(df["understat_xg"])
    return bundle, df


# ── Aggregations ───────────────────────────────────────────────────────────────
def possession_xg(vals):
    p = 1.0
    for v in vals: p *= (1.0 - float(v))
    return 1.0 - p


def build_team_table(df):
    # ── Attacking ──────────────────────────────────────────────────────────────
    poss = (df.groupby(["match_id", "team", "league"])
              .agg(shots=("xg","count"), goals=("goal","sum"),
                   xg_add=("xg","sum"), xg_poss=("xg", possession_xg))
              .reset_index())
    t = (poss.groupby(["team", "league"])
             .agg(matches        = ("match_id","nunique"),
                  shots          = ("shots","sum"),
                  goals          = ("goals","sum"),
                  xg             = ("xg_add","sum"),       # additive (industry standard)
                  xg_poss        = ("xg_poss","sum"),      # possession-adjusted
                  possessions    = ("match_id","count"),   # number of attacking possessions
             )
             .reset_index())

    # ── Defensive (flip team/opponent) ─────────────────────────────────────────
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

    t = t.merge(ta, on=["team","league"], how="left")

    # ── Derived columns ────────────────────────────────────────────────────────
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
    """
    Rate players 0–100 based on xG efficiency, shot quality and volume.
    Only includes players with >= min_goals in a season.

    Components (all normalized 0–1 before blending):
      efficiency  = goals / xG, capped at 2.0            weight 0.40
      quality     = xG / shot (mean chance quality)       weight 0.35
      volume      = log(shots) / log(max_shots)           weight 0.25
    """
    import numpy as np

    p = (df.groupby(["player", "team", "league", "season"])
           .agg(shots      = ("xg",   "count"),
                goals      = ("goal", "sum"),
                xg         = ("xg",   "sum"),
                xg_per_shot= ("xg",   "mean"))
           .reset_index())

    # Min goals filter
    p = p[p["goals"] >= min_goals].copy()
    if p.empty:
        return p

    p["xg"]          = p["xg"].clip(lower=0.01)   # avoid div/0
    p["xg_per_shot"] = p["xg_per_shot"].round(3)
    p["xg_diff"]     = (p["goals"] - p["xg"]).round(2)

    # ── Components ────────────────────────────────────────────────────────────
    # 1. Efficiency — goals/xG, capped at 2.0 (above this is noise)
    p["efficiency"] = (p["goals"] / p["xg"]).clip(upper=2.0)

    # 2. Shot quality — xG/shot, capped at 0.5 (penalties skew above this)
    p["quality"] = p["xg_per_shot"].clip(upper=0.5)

    # 3. Volume — log-scaled shots so Salah (200 shots) isn't 10x a 20-shot player
    p["volume"] = np.log1p(p["shots"]) / np.log1p(p["shots"].max())

    # ── Normalize each component to 0–1 ───────────────────────────────────────
    def norm(col):
        mn, mx = col.min(), col.max()
        if mx == mn: return pd.Series(0.5, index=col.index)
        return (col - mn) / (mx - mn)

    p["eff_n"] = norm(p["efficiency"])
    p["qlt_n"] = norm(p["quality"])
    p["vol_n"] = norm(p["volume"])

    # ── Blend ─────────────────────────────────────────────────────────────────
    p["rating"] = (
        p["eff_n"] * 0.40 +
        p["qlt_n"] * 0.35 +
        p["vol_n"] * 0.25
    )

    # Scale to 0–100
    p["rating"] = (norm(p["rating"]) * 100).round(1)

    # Drop intermediate columns
    p = p.drop(columns=["efficiency","quality","volume","eff_n","qlt_n","vol_n"])

    return p.sort_values("rating", ascending=False).reset_index(drop=True)


def build_match_timeline(df):
    rows = []
    for mid, grp in df.groupby("match_id"):
        teams = grp["team"].unique()
        if len(teams) < 2: continue
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
        if len(teams) < 2: continue
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


# ── Master refresh ─────────────────────────────────────────────────────────────
import gc

def refresh(seasons=None):
    print(f"\n[{datetime.now():%H:%M:%S}] Refreshing…")
    df = engineer(pull_shots(seasons))
    bundle, df = train(df)
    df.to_parquet("data/shots.parquet", index=False)
    with open("model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    del bundle
    gc.collect()
    build_team_table(df).to_parquet("data/teams.parquet", index=False)
    build_player_table(df).to_parquet("data/players.parquet", index=False)
    build_watchlist(df).to_parquet("data/watchlist.parquet", index=False)
    build_match_timeline(df).to_parquet("data/timeline.parquet", index=False)
    build_matches(df).to_parquet("data/matches.parquet", index=False)
    with open("data/last_updated.txt", "w") as f:
        f.write(datetime.now().isoformat())
    print("  Refresh complete ✓")


if __name__ == "__main__":
    import sys
    if "--rebuild-tables" in sys.argv:
        # Skip fetch — just reload shots.parquet and rebuild all derived tables
        print("Rebuilding tables from existing shots.parquet…")
        df = pd.read_parquet("data/shots.parquet")
        _, df = train(df)
        build_team_table(df).to_parquet("data/teams.parquet", index=False)
        build_player_table(df).to_parquet("data/players.parquet", index=False)
        build_watchlist(df).to_parquet("data/watchlist.parquet", index=False)
        build_match_timeline(df).to_parquet("data/timeline.parquet", index=False)
        build_matches(df).to_parquet("data/matches.parquet", index=False)
        with open("data/last_updated.txt", "w") as f:
            f.write(datetime.now().isoformat())
        print("Done ✓")
    elif "--current-season-only" in sys.argv:
        # Fetch current season, merge with existing shots, rebuild tables
        print(f"Refreshing current season ({CURRENT_SEASON}) only…")
        new_df = engineer(pull_shots([CURRENT_SEASON]))
        old_df = pd.read_parquet("data/shots.parquet")
        old_df = old_df[old_df["season"] != int(CURRENT_SEASON)]
        df = pd.concat([old_df, new_df], ignore_index=True)
        del old_df, new_df
        bundle, df = train(df)
        df.to_parquet("data/shots.parquet", index=False)
        build_team_table(df).to_parquet("data/teams.parquet", index=False)
        build_player_table(df).to_parquet("data/players.parquet", index=False)
        build_watchlist(df).to_parquet("data/watchlist.parquet", index=False)
        build_match_timeline(df).to_parquet("data/timeline.parquet", index=False)
        build_matches(df).to_parquet("data/matches.parquet", index=False)
        with open("data/last_updated.txt", "w") as f:
            f.write(datetime.now().isoformat())
        print("Done ✓")
    else:
        refresh()