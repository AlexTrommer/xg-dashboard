"""
train.py — xG model with per-situation specialist models
         Each situation group gets its own feature set + independent grid search.

Train set : 2020–2024
Test set  : 2025 (holdout)

Run:  python train.py
      python train.py --quick   (smaller grid, faster)
"""
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

Path("data").mkdir(exist_ok=True)

PENALTY_XG     = 0.76
GOAL_X         = 100.0
GOAL_Y_CENTER  = 50.0
GOAL_WIDTH_PCT = 11.6

# ── Per-situation configs ──────────────────────────────────────────────────────
# Each group has:
#   situations : list of raw situation strings from Understat
#   features   : tailored feature set
#   grid_hint  : param overrides on top of base grid
#
# OpenPlay   — full feature set, deep trees allowed
# FromCorner — almost exclusively headers; header/angle features dominate
# SetPiece   — position + shot type; no counter/fast-break noise
# Counter    — distance + angle + fast-break space proxies
SITUATION_CONFIGS = {
    "OpenPlay": {
        "situations": ["OpenPlay"],
        "features": [
            "distance", "angle_deg", "distance_sq", "distance_x_angle",
            "dist_x_angle_sq",
            "is_header", "is_set_piece", "is_rebound", "is_cross", "is_throughball",
            "header_x_dist", "header_x_angle", "cross_x_dist",
            "throughball_x_dist", "rebound_x_dist",
            "shot_zone", "is_big_chance", "minute_norm", "central_threat",
            "fast_break", "assisted_header", "is_cutback",
            "is_penalty_area", "weak_angle_header",
        ],
        "grid_hint": {
            "max_depth": [3, 4, 5],
        },
    },
    "FromCorner": {
        "situations": ["FromCorner"],
        "features": [
            "distance", "angle_deg", "distance_sq",
            "is_header", "header_x_dist", "header_x_angle",
            "shot_zone", "is_big_chance", "central_threat",
            "is_penalty_area", "weak_angle_header", "assisted_header",
            "minute_norm",
        ],
        "grid_hint": {
            "max_depth": [3, 4],
            "min_child_weight": [15, 30],
        },
    },
    "SetPiece": {
        # Understat uses "SetPiece" (indirect FK / crosses from FK) and
        # "DirectFreekick" (shots direct from a free kick) as separate strings
        "situations": ["SetPiece", "DirectFreekick"],
        "features": [
            "distance", "angle_deg", "distance_sq", "distance_x_angle",
            "is_header", "header_x_dist", "header_x_angle",
            "shot_zone", "is_big_chance", "central_threat",
            "is_penalty_area", "weak_angle_header",
            "minute_norm",
        ],
        "grid_hint": {
            "max_depth": [3, 4],
            "min_child_weight": [15, 25],
        },
    },
    # Note: Understat has no "FromCounter" situation — counters are tagged as
    # OpenPlay. The fast_break feature inside the OpenPlay specialist captures
    # counter-attack context instead.
}

FEATURES_GLOBAL = [
    "distance", "angle_deg", "distance_sq", "distance_x_angle",
    "dist_x_angle_sq",
    "is_header", "is_set_piece", "is_rebound", "is_cross", "is_throughball",
    "header_x_dist", "header_x_angle", "cross_x_dist",
    "throughball_x_dist", "rebound_x_dist",
    "shot_zone", "is_big_chance", "minute_norm", "central_threat",
    "fast_break", "assisted_header", "is_cutback",
    "is_penalty_area", "weak_angle_header",
]

MIN_SHOTS = 2000

# ── Feature engineering ────────────────────────────────────────────────────────
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
    df["is_counter"]     = pd.Series(0, index=df.index)  # Understat doesn't tag counters
    df["is_set_piece"]   = sit.str.contains("freekick|corner|setpiece").astype(int)
    df["is_penalty"]     = sit.str.contains("penalty").astype(int)
    df["is_cross"]       = la.str.contains("cross").astype(int)
    df["is_throughball"] = la.str.contains("throughball|through ball").astype(int)
    df["is_rebound"]     = la.str.contains("rebound|saves|blockedshot").astype(int)

    df["header_x_dist"]      = df["is_header"]      * df["distance"]
    df["header_x_angle"]     = df["is_header"]      * df["angle_deg"]
    df["cross_x_dist"]       = df["is_cross"]       * df["distance"]
    df["throughball_x_dist"] = df["is_throughball"] * df["distance"]
    df["rebound_x_dist"]     = df["is_rebound"]     * df["distance"]

    df["is_big_chance"]   = ((df["distance"] < 12) & (df["angle_deg"] > 20)).astype(int)
    df["central_threat"]  = 1.0 - (df["y"] - 50.0).abs() / 50.0
    df["is_penalty_area"] = ((df["x"] >= 83) & (df["y"] >= 22) & (df["y"] <= 78)).astype(int)

    def zone(r):
        if r["x"] >= 95: return 0
        if r["x"] >= 83 and 30 <= r["y"] <= 70: return 1
        return 2
    df["shot_zone"]   = df.apply(zone, axis=1)
    df["minute_norm"] = (df["minute"].fillna(45) / 120).clip(0, 1)

    transition_actions = {"TakeOn", "BallRecovery", "Throughball", "throughball"}
    df["fast_break"] = (
        (df["situation"] == "OpenPlay") &
        (la.isin(transition_actions) | la.str.contains("throughball", case=False, na=False))
    ).astype(int)
    df["assisted_header"]   = (df["is_header"] & df["is_cross"]).astype(int)
    df["is_cutback"]        = (
        (df["is_cross"] == 1) & (df["x"] > 88) &
        ((df["y"] < 28) | (df["y"] > 72))
    ).astype(int)
    df["weak_angle_header"] = (df["is_header"] & (df["angle_deg"] < 10)).astype(int)

    return df


# ── Load & prepare ─────────────────────────────────────────────────────────────
print("Loading data...")
df_raw = pd.read_parquet("data/shots.parquet")
df     = engineer(df_raw)
df     = df.dropna(subset=FEATURES_GLOBAL + ["goal"])

train_df = df[df["season"] != "2025"]
test_df  = df[df["season"] == "2025"]

train_np = train_df[train_df["is_penalty"] == 0]
test_np  = test_df[test_df["is_penalty"] == 0]

X_tr_g = train_np[FEATURES_GLOBAL].values
y_tr   = train_np["goal"].values
y_te   = test_np["goal"].values
und_te = test_np["understat_xg"].values

neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
spw_global = round(neg / pos, 2)

print(f"Train: {len(X_tr_g):,} shots ({int(y_tr.sum()):,} goals) -- 2020-2024")
print(f"Test:  {len(y_te):,} shots ({int(y_te.sum()):,} goals) -- 2025 holdout")
print()

print("Understat baseline on holdout:")
print(f"  ROC-AUC : {roc_auc_score(y_te, und_te):.4f}")
print(f"  Brier   : {brier_score_loss(y_te, und_te):.4f}")
print(f"  Log-loss: {log_loss(y_te, und_te):.4f}")
print()

# ── Param grids ────────────────────────────────────────────────────────────────
QUICK = "--quick" in sys.argv

BASE_GRID_FULL = {
    "n_estimators"    : [400, 700, 1000],
    "max_depth"       : [3, 4, 5],
    "learning_rate"   : [0.01, 0.03, 0.05],
    "min_child_weight": [10, 20, 30],
    "subsample"       : [0.70, 0.80],
    "colsample_bytree": [0.70, 0.80],
}
BASE_GRID_QUICK = {
    "n_estimators"    : [400, 800],
    "max_depth"       : [3, 4],
    "learning_rate"   : [0.02, 0.05],
    "min_child_weight": [10, 20],
    "subsample"       : [0.75],
    "colsample_bytree": [0.75],
}
BASE_GRID = BASE_GRID_QUICK if QUICK else BASE_GRID_FULL
MAX_GRID_ROWS = 300_000
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def run_grid(X, y, grid_overrides=None):
    grid = {**BASE_GRID}
    if grid_overrides:
        for k, v in grid_overrides.items():
            grid[k] = v
    n_combos = 1
    for v in grid.values(): n_combos *= len(v)

    neg_l, pos_l = (y == 0).sum(), (y == 1).sum()
    spw_l = round(neg_l / pos_l, 2)
    base = XGBClassifier(
        gamma=0.2, reg_alpha=0.2, reg_lambda=1.5,
        scale_pos_weight=spw_l, eval_metric="logloss",
        random_state=42, n_jobs=2, tree_method="hist",
    )
    if len(X) > MAX_GRID_ROWS:
        idx = np.random.RandomState(42).choice(len(X), MAX_GRID_ROWS, replace=False)
        Xs, ys = X[idx], y[idx]
    else:
        Xs, ys = X, y

    print(f"  {n_combos} combos x 3-fold on {len(Xs):,} shots...", end=" ", flush=True)
    t0 = datetime.now()
    search = GridSearchCV(
        estimator=base, param_grid=grid,
        scoring="neg_brier_score", cv=cv,
        verbose=0, n_jobs=1, refit=True,
    )
    search.fit(Xs, ys)
    elapsed = (datetime.now() - t0).seconds
    print(f"done in {elapsed//60}m{elapsed%60}s | Brier {-search.best_score_:.4f}")
    print(f"  Best: {search.best_params_}")
    return search.best_params_


def train_specialist(X, y, best_params):
    neg_l, pos_l = (y == 0).sum(), (y == 1).sum()
    spw_l = round(neg_l / pos_l, 2)
    base = XGBClassifier(
        **best_params,
        gamma=0.2, reg_alpha=0.2, reg_lambda=1.5,
        scale_pos_weight=spw_l, eval_metric="logloss",
        random_state=42, n_jobs=2, tree_method="hist",
    )
    m = CalibratedClassifierCV(base, method="isotonic", cv=5)
    m.fit(X, y)
    return m


# ── Phase 1: Global model ──────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1 -- Global fallback model")
print("=" * 60)
global_best = run_grid(X_tr_g, y_tr)

with open("data/best_params.json", "w") as f:
    json.dump(global_best, f, indent=2)

print("  Training global model on full training set...")
global_model = train_specialist(X_tr_g, y_tr, global_best)
models       = {"global": global_model}
feature_map  = {"global": FEATURES_GLOBAL}
situation_router = {}
all_best_params  = {"global": global_best}

# ── Phase 2: Per-situation specialists ────────────────────────────────────────
print()
print("=" * 60)
print("PHASE 2 -- Per-situation specialists")
print("=" * 60)

for group, cfg in SITUATION_CONFIGS.items():
    situations = cfg["situations"]
    features   = cfg["features"]
    grid_hint  = cfg.get("grid_hint", {})

    mask_tr = train_np["situation"].isin(situations)
    sub_tr  = train_np[mask_tr]
    sub_X   = sub_tr[features].values
    sub_y   = sub_tr["goal"].values

    print(f"\n-- {group} | {len(features)} features | {len(sub_X):,} shots --")

    if len(sub_X) < MIN_SHOTS or sub_y.sum() == 0:
        print(f"  Insufficient data -> routing to global")
        for s in situations: situation_router[s] = "global"
        continue

    best = run_grid(sub_X, sub_y, grid_overrides=grid_hint)
    all_best_params[group] = best

    print(f"  Training specialist...", end=" ", flush=True)
    t0 = datetime.now()
    m = train_specialist(sub_X, sub_y, best)
    elapsed = (datetime.now() - t0).seconds
    print(f"done in {elapsed//60}m{elapsed%60}s")

    models[group]      = m
    feature_map[group] = features
    for s in situations: situation_router[s] = group

    # Per-group holdout eval
    mask_te = test_np["situation"].isin(situations)
    if mask_te.sum() > 100:
        te_X = test_np.loc[mask_te, features].values
        te_y = test_np.loc[mask_te, "goal"].values
        te_u = test_np.loc[mask_te, "understat_xg"].values
        pp   = m.predict_proba(te_X)[:, 1]
        print(f"  Holdout ({mask_te.sum():,}): "
              f"AUC {roc_auc_score(te_y,pp):.4f} (ours) "
              f"vs {roc_auc_score(te_y,te_u):.4f} (understat) | "
              f"Brier {brier_score_loss(te_y,pp):.4f}")

with open("data/best_params_by_situation.json", "w") as f:
    json.dump(all_best_params, f, indent=2)
print("\nPer-situation params saved -> data/best_params_by_situation.json")


# ── Routing ────────────────────────────────────────────────────────────────────
def route_predict(df_input):
    preds = np.full(len(df_input), np.nan)
    df_input = df_input.reset_index(drop=True)
    df_input["_model_key"] = df_input["situation"].map(situation_router).fillna("global")
    for key, grp in df_input.groupby("_model_key"):
        m    = models.get(key, models["global"])
        feat = feature_map.get(key, FEATURES_GLOBAL)
        valid = grp.dropna(subset=feat)
        if len(valid):
            preds[valid.index] = m.predict_proba(valid[feat].values)[:, 1]
    return preds


# ── Full holdout eval ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print("HOLDOUT -- 2025 season (routed model)")
print("=" * 60)
preds      = route_predict(test_np.copy())
valid_mask = ~np.isnan(preds)
preds_v    = preds[valid_mask]
y_te_v     = y_te[valid_mask]
und_te_v   = und_te[valid_mask]

print(f"{'Metric':<14} {'Ours':>10} {'Understat':>12}")
print("-" * 38)
print(f"{'ROC-AUC':<14} {roc_auc_score(y_te_v,preds_v):>10.4f} {roc_auc_score(y_te_v,und_te_v):>12.4f}")
print(f"{'Brier':<14} {brier_score_loss(y_te_v,preds_v):>10.4f} {brier_score_loss(y_te_v,und_te_v):>12.4f}")
print(f"{'Log-loss':<14} {log_loss(y_te_v,preds_v):>10.4f} {log_loss(y_te_v,und_te_v):>12.4f}")

print()
print(f"{'Situation':<14} {'N':>7} {'AUC-ours':>10} {'AUC-und':>10} {'Brier':>8} {'Model':>12}")
print("-" * 65)
for group, cfg in SITUATION_CONFIGS.items():
    mask = test_np["situation"].isin(cfg["situations"])
    if mask.sum() < 50: continue
    feat = feature_map.get(group, FEATURES_GLOBAL)
    m    = models.get(group, models["global"])
    te_X = test_np.loc[mask, feat].values
    te_y = test_np.loc[mask, "goal"].values
    te_u = test_np.loc[mask, "understat_xg"].values
    pp   = m.predict_proba(te_X)[:, 1]
    used = group if group in models else "global"
    print(f"  {group:<12} {mask.sum():>7,} {roc_auc_score(te_y,pp):>10.4f} "
          f"{roc_auc_score(te_y,te_u):>10.4f} {brier_score_loss(te_y,pp):>8.4f} {used:>12}")

# ── Calibration ────────────────────────────────────────────────────────────────
print()
tmp = test_np.copy()
tmp["our_xg"] = preds
tmp = tmp[valid_mask]
tmp["xg_bucket"] = pd.cut(tmp["our_xg"],
    bins=[0,.05,.1,.2,.3,.5,1.01], include_lowest=True)
cal = tmp.groupby("xg_bucket", observed=False).agg(
    shots  =("goal","count"), actual=("goal","mean"),
    our_xg =("our_xg","mean"), und_xg=("understat_xg","mean"),
).reset_index()
cal["our_err"] = (cal["actual"] - cal["our_xg"]).round(4)
cal["und_err"] = (cal["actual"] - cal["und_xg"]).round(4)
print("CALIBRATION TABLE")
print(cal[["xg_bucket","shots","actual","our_xg","our_err","und_xg","und_err"]].to_string(index=False))

# ── Team diff ──────────────────────────────────────────────────────────────────
print()
test_all  = test_df.copy()
valid_all = test_all[test_all["is_penalty"] == 0].dropna(subset=FEATURES_GLOBAL)
routed    = route_predict(valid_all.copy())
test_all.loc[valid_all.index, "our_xg"] = routed
test_all.loc[test_all["is_penalty"] == 1, "our_xg"] = PENALTY_XG
test_all["our_xg"] = test_all["our_xg"].fillna(test_all["understat_xg"])
teams = test_all.groupby("team").agg(
    shots =("our_xg","count"), goals=("goal","sum"),
    our_xg=("our_xg","sum"),   und_xg=("understat_xg","sum"),
).reset_index()
teams["our_diff"] = (teams["goals"] - teams["our_xg"]).round(1)
teams = teams[teams["shots"] >= 50].sort_values("our_diff", ascending=False)
print("TEAM xG DIFF -- 2025 holdout")
print(teams.to_string(index=False))

# ── Save bundle ────────────────────────────────────────────────────────────────
bundle = {"models": models, "router": situation_router, "feature_map": feature_map}
with open("data/model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print()
print("=" * 60)
print(f"Saved -> data/model.pkl")
print(f"Specialists: {[k for k in models if k != 'global']}")
print(f"Global covers: {[s for s,k in situation_router.items() if k=='global']}")
print("Next: python data.py --rebuild-tables")
print("=" * 60)