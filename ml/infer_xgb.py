# ml/infer_xgb.py
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

from uknowuno.cards import Color, Card
from uknowuno.game_state import GameState
from ml.featurize import build_examples_for_legal_actions

# Absolute models dir
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

def _paths(n_players: int):
    mpath = MODEL_DIR / f"xgb_{n_players}p.json"
    meta  = MODEL_DIR / f"xgb_{n_players}p.meta.json"
    return mpath, meta  # return Paths, not strings

def load_xgb_for_players(n_players: int):
    import xgboost as xgb
    mpath, meta = _paths(n_players)

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"MODEL_DIR not found: {MODEL_DIR}")
    if not mpath.exists():
        raise FileNotFoundError(f"Model file missing: {mpath}")
    if not meta.exists():
        raise FileNotFoundError(f"Meta file missing: {meta}")

    with meta.open() as f:
        md = json.load(f)
    feature_dim = int(md["feature_dim"])

    booster = xgb.Booster()
    booster.load_model(str(mpath))  # xgboost expects a str path
    booster._expected_dim = feature_dim  # type: ignore[attr-defined]
    return booster

def predict_scores(model, X_mat: np.ndarray) -> np.ndarray:
    if hasattr(model, "inplace_predict"):
        return model.inplace_predict(X_mat)
    return model.predict(X_mat)  # fallback for sklearn wrappers

def pick_with_xgb(model, state: GameState, me: int):
    X, acts = build_examples_for_legal_actions(state, me)
    if not X:
        return None, None, []
    import numpy as _np
    X_mat = _np.asarray(X, dtype=_np.float32)

    exp = getattr(model, "_expected_dim", X_mat.shape[1])
    if X_mat.shape[1] != exp:
        raise ValueError(f"Feature shape mismatch, expected: {exp}, got: {X_mat.shape[1]}")

    scores = predict_scores(model, X_mat)
    i = int(_np.argmax(scores))
    card, color = acts[i]
    return card, color, list(zip(acts, scores.tolist()))
