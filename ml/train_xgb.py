# ml/train_xgb.py
from __future__ import annotations
import os, json, random, argparse
import numpy as np
from xgboost.callback import EarlyStopping


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from uknowuno.engine import start_game_with_my_hand, legal_moves_for_player
from uknowuno.cards import Card, Color, Rank
from uknowuno.game_state import GameState

from ml.featurize import build_examples_for_legal_actions
from ml.rollout_oracle import evaluate_current_position  # uses your simulate_to_end etc.

# -------------------------
# Args
# -------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--players", type=int, choices=[2, 3, 4], required=True)
ap.add_argument("--games", type=int, default=20000)
ap.add_argument("--worlds", type=int, default=16)      # kept for future; labels already averaged inside eval if you add worlds upstream
ap.add_argument("--rollouts", type=int, default=64)    # rollouts per action for labels
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", type=str, default=None)
args = ap.parse_args()

rng = random.Random(args.seed)

# -------------------------
# Start-state sampler
# -------------------------
def new_start_state() -> GameState:
    """Create a manual-mode starting state with a safe top and a playable hand."""
    top = Card(Color.RED, Rank.R4)
    # Hand with at least one legal play vs RED R4
    my_hand = [
        Card(Color.RED, Rank.R1), Card(Color.RED, Rank.R6),
        Card(Color.GREEN, Rank.R4), Card(Color.YELLOW, Rank.R8),
        Card(Color.BLUE, Rank.R2), Card(None, Rank.WILD),
        Card(Color.BLUE, Rank.REVERSE),
    ]
    rng.shuffle(my_hand)
    g = start_game_with_my_hand(
        num_players=args.players,
        my_hand=my_hand,
        my_index=0,
        seed=rng.randint(0, 2**31-1),
        hand_size=7,
        initial_top=top,
        initial_active_color=top.color,
        manual_mode=True,           # matches real-play usage
    )
    return g

# -------------------------
# Dataset collection
# -------------------------
def collect_dataset(n_games: int, rollouts_per_action: int, seed: int):
    X_all: list[np.ndarray] = []
    y_all: list[float]      = []

    local_rng = random.Random(seed)

    for gi in range(n_games):
        state = new_start_state()
        me = state.current_player

        # If there are no legal actions, skip (should be rare with our hand/top)
        if not legal_moves_for_player(state, me):
            continue

        # Build (state,action) feature vectors for each legal action
        X_actions, acts = build_examples_for_legal_actions(state, me)

        # Label with rollout win-rates for the same action order
        ests = evaluate_current_position(
            world=state,
            my_id_world=me,
            n_rollouts_per_action=rollouts_per_action,
            rng_seed=local_rng.randint(0, 2**31-1),
        )

        # Map (card, chosen_color) → win_rate so order lines up
        label_map = { (e.card.short(), e.chosen_color.name if e.chosen_color else None): e.win_rate
                      for e in ests }

        for (card, color), feats in zip(acts, X_actions):
            key = (card.short(), color.name if color else None)
            y = label_map.get(key, None)
            if y is None:
                # If an action wasn't evaluated due to a rare mismatch, skip it.
                continue
            X_all.append(feats.astype(np.float32))
            y_all.append(float(y))

        if (gi+1) % 500 == 0:
            print(f"[collect] {gi+1}/{n_games} examples so far: {len(y_all)}")

    if not X_all:
        raise RuntimeError("No training data collected. Check sampler/legality.")

    X = np.vstack(X_all)
    y = np.array(y_all, dtype=np.float32)
    return X, y

# -------------------------
# Train
# -------------------------
def main():
    print(f"Collecting data: players={args.players}, games={args.games}, rollouts={args.rollouts}")
    X, y = collect_dataset(args.games, args.rollouts, args.seed)
    print("Dataset:", X.shape, y.shape)

    # Split (80/10/10)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.20, random_state=args.seed)
    X_val,   X_test, y_val,   y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=args.seed)

    model = XGBRegressor(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=-1,
        tree_method="hist",
    )

    model.set_params(eval_metric="rmse")


    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluate a quick test metric (not critical, just a sanity check)
    test_pred = model.predict(X_test)
    mae = float(np.mean(np.abs(test_pred - y_test)))
    print(f"Test MAE vs oracle labels: {mae:.4f}")

    # Save model + meta (save underlying Booster to avoid sklearn wrapper bug)
    import os, json
    os.makedirs("models", exist_ok=True)
    outpath = args.out or f"models/xgb_{args.players}p.json"

    booster = model.get_booster()
    booster.save_model(outpath)  # <-- key change

    meta = {
        "players": args.players,
        "feature_dim": int(X.shape[1]),
        "featurizer_version": "v1",
        "format": "booster"   # mark how we saved it
    }
    with open(outpath.replace(".json", ".meta.json"), "w") as f:
        json.dump(meta, f)

    print("Saved:", outpath)
    print("Feature dim:", X.shape[1])

    print("Feature dim:", X.shape[1])

if __name__ == "__main__":
    main()
