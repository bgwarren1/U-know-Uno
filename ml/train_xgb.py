# ml/train_xgb.py
from __future__ import annotations
import argparse, random, os
from typing import List, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from random import choice

from uknowuno.cards import Card, Color
from uknowuno.cards import Rank 
from uknowuno.engine import start_game_with_my_hand
from uknowuno.rules import full_deck
from ml.featurize import build_examples_for_legal_actions
from ml.rollout_oracle import evaluate_ensemble

ART_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

def sample_start_state(num_players=None, my_index=0, seed=0):
    rng = random.Random(seed)
    if num_players is None:
        num_players = choice([2, 3, 4])  # train on mixed sizes

    deck = full_deck()
    rng.shuffle(deck)
    my_hand = [deck.pop() for _ in range(7)]

    # pick a valid initial top, avoid WILD_DRAW4 and conflict rule if your engine enforces it
    forbidden = {(c.color, c.rank) for c in my_hand}
    initial_top = None
    while deck:
        cand = deck.pop()
        if cand.rank == Rank.WILD_DRAW4:
            continue
        if (cand.color, cand.rank) not in forbidden:
            initial_top = cand
            break
    if initial_top is None:
        # rare fallback: resample entirely
        return sample_start_state(num_players=num_players, my_index=my_index, seed=rng.randint(0, 2**31-1))

    names = [f"Player {i}" for i in range(num_players)]
    return start_game_with_my_hand(
        num_players=num_players,
        my_hand=my_hand,
        my_index=my_index,
        names=names,
        seed=seed,
        hand_size=7,
        initial_top=initial_top,
        initial_active_color=initial_top.color if not initial_top.is_wild() else None,
        manual_mode=False,
    )


def _act_key(card: Card, chosen_color: Color | None) -> tuple:
    """Key used both when reading oracle results and when aligning features."""
    return (
        card.rank.name,
        card.color.name if card.color is not None else None,
        chosen_color.name if chosen_color is not None else None,
    )


def collect_dataset(
    games: int,
    worlds: int,
    rollouts: int,
    base_seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[tuple]]:
    """
    Build a supervised dataset for XGBoost:
      - X: features for (state, legal action)
      - y: oracle-estimated win rate for that action
      - meta: lightweight info per row (optional for debugging)
    Assumes `sample_start_state(...)` samples 2/3/4-player games uniformly.
    """
    rng = random.Random(base_seed)

    X_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    meta: List[tuple] = []  # e.g., [(card, chosen_color, num_players, hand_len), ...]

    for g in range(games):
        # Retry a few times in case start generation throws (e.g., conflict rules)
        for _ in range(5):
            try:
                state = sample_start_state(seed=rng.randint(0, 2**31 - 1))
                break
            except ValueError:
                continue
        else:
            # couldn't get a valid start; skip this sample
            continue

        # Label with oracle (averaged across determinized worlds)
        ests = evaluate_ensemble(
            state,
            my_id=state.my_index,
            n_worlds=worlds,
            n_rollouts_per_action=rollouts,
            rng_seed=rng.randint(0, 2**31 - 1),
            force_determinize=False,  # non-manual states already have a real deck
        )
        # Map oracle outputs for quick lookup
        est_map = { _act_key(e.card, e.chosen_color): float(e.win_rate) for e in ests }

        # Build features for each legal action (wilds expanded to 4 chosen-color options)
        X_list, acts = build_examples_for_legal_actions(state, state.my_index)
        if not X_list:
            continue

        for x, (card, chosen_color) in zip(X_list, acts):
            key = _act_key(card, chosen_color)
            y = est_map.get(key, None)
            if y is None:
                # In rare mismatches (shouldn't happen), skip this row
                continue

            X_rows.append(x.astype(np.float32, copy=False))
            y_rows.append(y)
            meta.append((card, chosen_color, state.num_players(), len(state.players[state.my_index].hand)))

    if not X_rows:
        # Guardrail: avoid cryptic errors downstream
        raise RuntimeError("No training rows collected — check rollout settings or featurizer consistency.")

    X = np.ascontiguousarray(np.vstack(X_rows), dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    return X, y, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200, help="number of sampled start states")
    ap.add_argument("--worlds", type=int, default=8, help="determinized worlds per position")
    ap.add_argument("--rollouts", type=int, default=32, help="rollouts per action inside each world")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    print(f"Collecting data: games={args.games}, worlds={args.worlds}, rollouts={args.rollouts}")
    X, y, _ = collect_dataset(args.games, args.worlds, args.rollouts, args.seed)
    print("Dataset:", X.shape, y.shape)

    model = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=4,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=args.seed,
    )
    model.fit(X, y)
    outpath = os.path.join(ART_DIR, "xgb_uno.json")
    model.get_booster().save_model(outpath)
    print("Saved booster to", outpath)

if __name__ == "__main__":
    main()
