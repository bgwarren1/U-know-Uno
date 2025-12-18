# ml/infer_xgb.py
from __future__ import annotations
import os
import numpy as np
from xgboost import XGBRegressor

from uknowuno.engine import legal_moves_for_player, play_card_by_index
from uknowuno.cards import Color, Card
from uknowuno.game_state import GameState
from ml.featurize import build_examples_for_legal_actions

ART_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

def load_xgb(path: str | None = None) -> XGBRegressor:
    path = path or os.path.join(ART_DIR, "xgb_uno.json")
    model = XGBRegressor()
    model.load_model(path)
    return model

def pick_with_xgb(model: XGBRegressor, state: GameState, me: int):
    X, acts = build_examples_for_legal_actions(state, me)
    if not X:
        return None, None, []
    X_mat = np.vstack(X)
    scores = model.predict(X_mat)
    i = int(np.argmax(scores))
    return acts[i][0], acts[i][1], list(zip(acts, scores.tolist()))
