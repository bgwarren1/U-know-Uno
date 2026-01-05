# ml/infer_xgb.py
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
import os
from uknowuno.cards import Color, Card
from uknowuno.game_state import GameState
from ml.featurize import build_examples_for_legal_actions

# Models live in ../models/xgb_{2,3,4}p.json with a sidecar .meta.json
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


def _paths(n_players: int) -> tuple[Path, Path]:
    mpath = MODEL_DIR / f"xgb_{n_players}p.json"
    meta  = MODEL_DIR / f"xgb_{n_players}p.meta.json"
    return mpath, meta

def load_xgb_for_players(n_players: int):
    """
    Load a Booster saved by train_xgb.py and attach the expected feature_dim
    from the sidecar meta file (used to guard against feature-shape drift).
    """
    # Lazy import keeps module import safe even if xgboost isn't installed yet
    import xgboost as xgb

    mpath, meta = _paths(n_players)
    if not mpath.exists():
        raise FileNotFoundError(f"Model file not found for {n_players} players: {mpath}")
    if not meta.exists():
        raise FileNotFoundError(f"Sidecar meta not found for {n_players} players: {meta}")

    with meta.open() as f:
        md = json.load(f)
    feature_dim = int(md["feature_dim"])

    booster = xgb.Booster()
    booster.load_model(str(mpath))
    # attach for runtime shape check
    booster._expected_dim = feature_dim  # type: ignore[attr-defined]
    return booster

def load_xgb(path: str | Path | None = None):
    """
    Convenience loader if you have a single model path.
    Expects a Booster file. If you need meta enforcement, prefer load_xgb_for_players.
    """
    # Lazy import here too
    import xgboost as xgb

    if path is None:
        # Fallback default—adjust if you keep a single canonical model
        path = MODEL_DIR / "xgb_3p.json"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster

def predict_scores(model, X_mat: np.ndarray) -> np.ndarray:
    """
    Works for both Booster (preferred) and sklearn wrapper, if you ever use it.
    """
    if hasattr(model, "inplace_predict"):
        return model.inplace_predict(X_mat)
    # Fallback for sklearn wrappers
    try:
        return model.predict(X_mat)
    except Exception as e:
        raise RuntimeError(f"Unsupported model type for prediction: {type(model)}") from e

def pick_with_xgb(model, state: GameState, me: int):
    """
    Build features for each legal action, run model to score them,
    and return (best_card, chosen_color, [(action, score), ...]).
    """
    X, acts = build_examples_for_legal_actions(state, me)
    if not X:
        return None, None, []

    # XGBoost likes float32 here; also keeps inplace_predict fast.
    X_mat = np.asarray(X, dtype=np.float32)

    # If the model came from load_xgb_for_players, this will be present
    exp = getattr(model, "_expected_dim", X_mat.shape[1])
    if X_mat.shape[1] != exp:
        raise ValueError(f"Feature shape mismatch, expected: {exp}, got: {X_mat.shape[1]}")

    scores = predict_scores(model, X_mat)
    i = int(np.argmax(scores))
    (card, color) = acts[i]
    return card, color, list(zip(acts, scores.tolist()))
