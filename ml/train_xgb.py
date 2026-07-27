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
from uknowuno.rules import full_deck

from ml.featurize import build_examples_for_legal_actions
from ml.rollout_oracle import evaluate_ensemble  # determinizes per world -> individual opponent hands

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

# -------------------------
# Mid-game position sampler
# -------------------------
def sample_midgame_position(rng: random.Random) -> GameState:
    """Sample a diverse mid-game snapshot for training (NOT a game start).

    In real use the app is opened at ANY point in a game, so the model is queried at every hand
    size, not just turn 1. The old sampler returned the SAME 7-card position every call, so
    training on 20000 games was really one position 20000 times. Instead we sample a *snapshot* of
    a game already in progress, randomizing:
      - the current top card (its color/rank -> the active color),
      - my current hand: contents AND size (1..7 cards remaining),
      - each opponent's remaining card count (1..7).

    Note: this is a snapshot, not a dealt game -- a real Uno game always starts at 7 cards each;
    here the smaller counts represent cards already played. My hand and the top are dealt from ONE
    shuffled deck so the position is always legal (no card exceeds its real deck count, which would
    otherwise crash determinization). We also require at least one legal move so the example isn't
    skipped by the collector. Opponent counts are drawn independently of my hand size, so the joint
    distribution is an approximation of real play, not an exact match (accepted tradeoff).
    """
    for _ in range(100):
        deck = full_deck()
        rng.shuffle(deck)

        # Current top: re-flip on a wild so the active color is well-defined (standard Uno rule).
        top = deck.pop()
        while top.is_wild():
            deck.insert(0, top)
            top = deck.pop()

        # My hand: random remaining size, dealt from the same deck so nothing collides with the top.
        hand_size = rng.randint(1, 7)
        my_hand = [deck.pop() for _ in range(hand_size)]

        g = start_game_with_my_hand(
            num_players=args.players,
            my_hand=my_hand,
            my_index=0,
            hand_size=hand_size,
            initial_top=top,
            initial_active_color=top.color,
            manual_mode=True,           # matches real-play usage
        )

        # Vary opponents' remaining counts to represent mid-game (old code left them all at 7).
        for pid in range(g.num_players()):
            if pid != g.my_index:
                g.players[pid].hidden_count = rng.randint(1, 7)

        # Keep only positions where I have a legal move (otherwise there's nothing to label).
        if legal_moves_for_player(g, g.current_player):
            return g

    # Fallback (practically never hit): return the last sampled snapshot.
    return g

# -------------------------
# Dataset collection
# -------------------------
def collect_dataset(n_games: int, rollouts_per_action: int, seed: int):
    X_all: list[np.ndarray] = []
    y_all: list[float]      = []

    local_rng = random.Random(seed)

    for gi in range(n_games):
        state = sample_midgame_position(local_rng)
        me = state.current_player

        # If there are no legal actions, skip (should be rare with our hand/top)
        if not legal_moves_for_player(state, me):
            continue

        # Build (state,action) feature vectors for each legal action
        X_actions, acts = build_examples_for_legal_actions(state, me)

        # Label with rollout win-rates for the same action order.
        # FIX (opponent-hands): use evaluate_ensemble with force_determinize so every training
        # rollout runs in a concrete world where opponents hold their OWN dealt hands. The old
        # evaluate_current_position call ran on the raw manual state (empty deck/pool), so
        # opponents were frozen and never played -- producing badly skewed labels.
        ests = evaluate_ensemble(
            state,
            my_id=me,
            n_worlds=args.worlds,
            n_rollouts_per_action=rollouts_per_action,
            rng_seed=local_rng.randint(0, 2**31-1),
            force_determinize=True,
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

if __name__ == "__main__":
    main()
