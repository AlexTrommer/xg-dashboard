"""
check.py — sanity check before the full overnight run
Tests each step with minimal data so you catch errors in ~2 minutes.

Run:  python check.py
"""
import sys
import traceback
import asyncio
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"

errors = []

def ok(msg):    print(f"{PASS} {msg}")
def fail(msg):  print(f"{FAIL} {msg}"); errors.append(msg)
def warn(msg):  print(f"{WARN} {msg}")

print("=" * 50)
print("  xG Dashboard — pre-flight check")
print("=" * 50)
print()

# ── 1. Imports ─────────────────────────────────────────────────────────────────
print("[1/6] Checking dependencies…")
deps = [
    ("xgboost",     "xgboost"),
    ("sklearn",     "scikit-learn"),
    ("pandas",      "pandas"),
    ("numpy",       "numpy"),
    ("fastapi",     "fastapi"),
    ("uvicorn",     "uvicorn"),
    ("understat",   "understat"),
    ("aiohttp",     "aiohttp"),
    ("pyarrow",     "pyarrow"),
    ("apscheduler", "apscheduler"),
]
for mod, pkg in deps:
    try:
        __import__(mod)
        ok(pkg)
    except ImportError:
        fail(f"{pkg} not installed — run: pip install {pkg}")
print()

# ── 2. data.py imports cleanly ─────────────────────────────────────────────────
print("[2/6] Importing data.py…")
try:
    import data as D
    ok(f"data.py loaded — {len(D.FEATURES)} features, {len(D.LEAGUES)} leagues")
except Exception as e:
    fail(f"data.py failed to import: {e}")
    traceback.print_exc()
    print("\nCannot continue — fix data.py first.")
    sys.exit(1)
print()

# ── 3. Fetch a single match from Understat ─────────────────────────────────────
print("[3/6] Testing Understat connection (fetching 1 season of EPL)…")
try:
    import understat, aiohttp

    async def _test_fetch():
        async with aiohttp.ClientSession() as session:
            u = understat.Understat(session)
            # Just grab one match to verify connectivity
            results = await u.get_league_results("EPL", "2024")
            if not results:
                return []
            mid = results[0].get("id")
            shots = await u.get_match_shots(mid)
            all_shots = []
            for side in ("h", "a"):
                all_shots.extend(shots.get(side, []))
            return all_shots

    shots = asyncio.run(_test_fetch())
    if shots:
        ok(f"Understat reachable — got {len(shots)} shots from first match")
    else:
        warn("Understat returned 0 shots — API may be slow or down")
except Exception as e:
    fail(f"Understat fetch failed: {e}")
print()

# ── 4. Feature engineering ─────────────────────────────────────────────────────
print("[4/6] Testing feature engineering…")
try:
    # Build a minimal fake shot dataframe
    fake = pd.DataFrame([{
        "id": "test_1", "season": "2024", "league": "Premier League",
        "match_id": "test_match", "player": "Test Player", "player_id": "1",
        "team": "Team A", "opponent": "Team B", "h_a": "h",
        "minute": 45, "x": 88.0, "y": 50.0, "goal": 1,
        "situation": "OpenPlay", "shot_type": "RightFoot",
        "last_action": "Cross", "understat_xg": 0.35, "match_date": "2024-01-01",
    }, {
        "id": "test_2", "season": "2024", "league": "Premier League",
        "match_id": "test_match", "player": "Test Player 2", "player_id": "2",
        "team": "Team B", "opponent": "Team A", "h_a": "a",
        "minute": 60, "x": 75.0, "y": 38.0, "goal": 0,
        "situation": "FromCorner", "shot_type": "Head",
        "last_action": "Cross", "understat_xg": 0.12, "match_date": "2024-01-01",
    }])

    eng = D.engineer(fake)

    missing = [f for f in D.FEATURES if f not in eng.columns]
    if missing:
        fail(f"Missing features after engineering: {missing}")
    else:
        ok(f"All {len(D.FEATURES)} features produced correctly")

    # Spot-check a few values
    row = eng.iloc[0]
    assert row["is_cross"] == 1,        "is_cross should be 1 for Cross last_action"
    assert row["is_header"] == 0,       "is_header should be 0 for RightFoot"
    assert row["distance"] > 0,         "distance should be positive"
    assert 0 <= row["angle_deg"] <= 90, "angle_deg out of range"
    assert row["central_threat"] == 1.0,"central shot should have central_threat=1"

    row2 = eng.iloc[1]
    assert row2["is_header"] == 1,      "is_header should be 1 for Head shot_type"
    assert row2["assisted_header"] == 1,"assisted_header should be 1 (header from cross)"

    ok("Feature value spot-checks passed")
except AssertionError as e:
    fail(f"Feature value wrong: {e}")
except Exception as e:
    fail(f"Engineer failed: {e}")
    traceback.print_exc()
print()

# ── 5. Model train + predict on fake data ─────────────────────────────────────
print("[5/6] Testing model training on synthetic data…")
try:
    from xgboost import XGBClassifier
    from sklearn.calibration import CalibratedClassifierCV

    # Generate enough fake rows to train (need >3 for CV)
    np.random.seed(42)
    n = 200
    fake_big = pd.DataFrame({
        "season": ["2023"] * 150 + ["2024"] * 50,
        "situation": np.random.choice(["OpenPlay","FromCorner","FreekickShot"], n),
        "shot_type": np.random.choice(["RightFoot","LeftFoot","Head"], n),
        "last_action": np.random.choice(["Pass","Cross","Rebound"], n),
        "minute": np.random.randint(1, 90, n),
        "x": np.random.uniform(60, 100, n),
        "y": np.random.uniform(20, 80, n),
        "goal": np.random.binomial(1, 0.1, n),
        "understat_xg": np.random.uniform(0.03, 0.5, n),
        "match_id": [f"m{i//4}" for i in range(n)],
    })
    fake_big["id"] = [f"s{i}" for i in range(n)]

    eng_big = D.engineer(fake_big)
    eng_big = eng_big.dropna(subset=D.FEATURES + ["goal"])
    eng_big["is_penalty"] = 0

    # Override cv to 3 for speed on small synthetic data
    from xgboost import XGBClassifier
    from sklearn.calibration import CalibratedClassifierCV
    _base = XGBClassifier(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1, eval_metric="logloss")
    _model = CalibratedClassifierCV(_base, method="isotonic", cv=3)
    eng_np = eng_big[eng_big["is_penalty"]==0].dropna(subset=D.FEATURES+["goal"])
    _model.fit(eng_np[D.FEATURES].values, eng_np["goal"].values)
    eng_big.loc[eng_np.index, "xg"] = _model.predict_proba(eng_np[D.FEATURES].values)[:,1]
    df_pred = eng_big

    assert "xg" in df_pred.columns, "xg column missing after train()"
    assert df_pred["xg"].between(0, 1).all(), "xg values out of [0,1]"
    ok(f"Model trained and predicted — xg range [{df_pred['xg'].min():.3f}, {df_pred['xg'].max():.3f}]")
except Exception as e:
    fail(f"Model train/predict failed: {e}")
    traceback.print_exc()
print()

# ── 6. Check existing data files (if any) ─────────────────────────────────────
print("[6/6] Checking existing data files…")
files = ["shots", "teams", "players", "timeline", "matches"]
found = []
for name in files:
    p = Path(f"data/{name}.parquet")
    if p.exists():
        df = pd.read_parquet(p)
        ok(f"data/{name}.parquet — {len(df):,} rows")
        found.append(name)
    else:
        warn(f"data/{name}.parquet not found — run python data.py")

if Path("data/model.pkl").exists():
    with open("data/model.pkl", "rb") as f:
        m = pickle.load(f)
    ok("data/model.pkl exists and loads cleanly")
else:
    warn("data/model.pkl not found — run python data.py")

if Path("data/best_params.json").exists():
    import json
    p = json.loads(Path("data/best_params.json").read_text())
    ok(f"data/best_params.json found — {p}")
else:
    warn("data/best_params.json not found — run python train.py to tune (optional)")

print()

# ── Summary ────────────────────────────────────────────────────────────────────
print("=" * 50)
if errors:
    print(f"  {len(errors)} error(s) found — fix before running:")
    for e in errors:
        print(f"    • {e}")
    sys.exit(1)
else:
    print("  All checks passed ✓")
    print()
    if len(found) == len(files):
        print("  Ready to run:  uvicorn api:app --reload --port 8000")
    else:
        print("  Ready to run:  python data.py")
print("=" * 50)