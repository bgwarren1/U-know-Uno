# ml/train_xgb.py
from __future__ import annotations
import argparse, random, os
from typing import List, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from uknowuno.cards import Card
from uknowuno.cards import Rank 
from uknowuno.engine import start_game_with_my_hand
from uknowuno.rules import full_deck
from ml.featurize import build_examples_for_legal_actions
from ml.rollout_oracle import evaluate_ensemble

ART_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

def sample_start_state(num_players=4, my_index=0, seed=0):
    rng = random.Random(seed)
    deck = full_deck()
    rng.shuffle(deck)

    # deal my hand
    my_hand = [deck.pop() for _ in range(7)]
    forbidden = {(c.color, c.rank) for c in my_hand}

    # pick a valid initial top: not conflicting with my hand, and avoid WILD_DRAW4
    initial_top = None
    while deck:
        cand = deck.pop()
        if cand.rank == Rank.WILD_DRAW4:
            continue
        if (cand.color, cand.rank) not in forbidden:
            initial_top = cand
            break

    if initial_top is None:
        # extremely unlikely; just resample entirely
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

def collect_dataset(games: int, worlds: int, rollouts: int, base_seed: int) -> Tuple[np.ndarray, np.ndarray, List[Tuple[Card, int]]]:
    X_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    meta: List[Tuple[Card, int]] = []  # (card, chosen_color_idx or -1)
    rng = random.Random(base_seed)

    for g in range(games):
        state = sample_start_state(seed=rng.randint(0, 2**31-1))
        # make sure it's my turn; at game start it is
        ests = evaluate_ensemble(
            state, my_id=state.my_index,
            n_worlds=worlds, n_rollouts_per_action=rollouts,
            rng_seed=rng.randint(0, 2**31-1),
            force_determinize=False
        )

        # build examples for current legal actions (wilds expanded by color)
        X, acts = build_examples_for_legal_actions(state, state.my_index)
        if not X: 
            continue

        # map estimates to our expanded (card, color) list
        # we keep best color for wilds; evaluate_ensemble already did that
        # so match on (rank, color, chosen_color)
        def key(card, color):
            return (card.rank.name, card.color.name if card.color else None, color.name if color else None)

        est_map = { key(e.card, e.chosen_color): e.win_rate for e in ests }
        for x, (card, chosen_color) in zip(X, acts):
            y = est_map.get(key(card, chosen_color), None)
            if y is None:
                # If this legal action wasn't evaluated (should be rare), skip
                continue
            X_rows.append(x.astype(np.float32))
            y_rows.append(float(y))
            color_idx = [-1, 0, 1, 2, 3][0]  # not used downstream; placeholder
            meta.append((card, -1))

    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=np.float32)
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
