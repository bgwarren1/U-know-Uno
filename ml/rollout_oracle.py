from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from uknowuno.cards import Card, Color, Rank
from uknowuno.engine import (
    legal_moves_for_player,
    play_card_by_index,
    opponent_play_from_menu,
    draw_for_player,
    pass_turn,
)
from uknowuno.game_state import GameState
from uknowuno.strategy import recommend_move
from uknowuno.rules import full_deck


# enumerate all 54 card types
def all_card_types() -> List[Card]:
    COLORS = [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]
    NUMS = [Rank[f"R{i}"] for i in range(10)]
    ACTIONS = [Rank.SKIP, Rank.REVERSE, Rank.DRAW2]
    WILDS = [Rank.WILD, Rank.WILD_DRAW4]
    types: List[Card] = [Card(c, r) for c in COLORS for r in (NUMS + ACTIONS)]
    types += [Card(None, r) for r in WILDS]
    return types





"""
HELPERS
"""

# Helper to check the oracle simulations are all individual ; 
# when we determinize a world or apply an action the player list can be rebuilt. If we keep the old my_id, you’ll credit wins to the wrong seat → tiny win%
def map_pid_by_name(src: "GameState", dst: "GameState", pid: int) -> int:
    """Return the index in `dst` that corresponds to `src.players[pid]` (via name)."""
    target = src.players[pid].name
    for j, p in enumerate(dst.players):
        if p.name == target:
            return j
    # Fallback to same index if names somehow changed.
    return min(pid, dst.num_players() - 1)

def estimate_baseline(world: "GameState", my_id_world: int, trials: int = 256) -> float:
    rng = random.Random(1337)
    finished = wins = 0
    for _ in range(trials):
        s = copy.deepcopy(world)
        done, w = simulate_to_end(s, my_id_world, rng, max_turns=300)
        if done:
            finished += 1
            wins += (w == my_id_world)
    return (wins / finished) if finished else 0.0



def find_hand_index_of_card(hand: List[Card], proto: Card) -> Optional[int]:
    """
    Return the index of the first card in `hand` that matches the type of `proto`.
    - For non-wilds: match both color and rank.
    - For wilds: match rank only (hand wilds have color=None).
    """
    for i, c in enumerate(hand):
        if proto.is_wild():
            if c.rank == proto.rank:
                return i
        else:
            if (c.rank == proto.rank) and (c.color == proto.color):
                return i
    return None


def my_best_color_from_hand(hand: List[Card]) -> Optional[Color]:
    """
    Simple heuristic: pick the color you have the most of (ignoring wilds).
    """
    counts = {Color.RED: 0, Color.YELLOW: 0, Color.GREEN: 0, Color.BLUE: 0}
    for c in hand:
        if c.color in counts:
            counts[c.color] += 1  # type: ignore
    # pick the color with max count; if all zero, return None
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else None

def _remove_one(pool, card):
    for i, c in enumerate(pool):
        if card.is_wild():
            if c.rank == card.rank:
                pool.pop(i); return True
        else:
            if c.rank == card.rank and c.color == card.color:
                pool.pop(i); return True
    return False

def determinize_from_counts(state: GameState, rng: random.Random) -> GameState:
    """Sample a full world (hidden hands + deck) consistent with your visible info."""
    s = copy.deepcopy(state)
    pool = full_deck()

    # remove your hand & entire discard (including top)
    for c in s.players[s.my_index].hand:
        assert _remove_one(pool, c)
    for c in s.discard:
        assert _remove_one(pool, c)

    # deal identities to opponents based on their hidden_count
    s.hidden_pool = []
    for pid in range(s.num_players()):
        if pid == s.my_index:
            continue
        k = s.players[pid].hidden_count
        draw = rng.sample(pool, k)
        for c in draw: _remove_one(pool, c)
        s.hidden_pool.extend(draw)

    rng.shuffle(pool)
    s.deck = pool
    s.manual_mode = False
    return s



def hidden_pool_has(pool: List[Card], proto: Card) -> bool:
    """
    Does the hidden_pool contain at least one card of this *type*?
    """
    for c in pool:
        if proto.is_wild():
            if c.rank == proto.rank:
                return True
        else:
            if (c.rank == proto.rank) and (c.color == proto.color):
                return True
    return False




"""
Opponent Policy
"""
@dataclass
class OpponentPolicyConfig:
    prefer_actions_first: bool = True
    prefer_keep_color: bool = True


def opponent_pick_card_for_menu(state: GameState, pid: int, rng: random.Random,
                                cfg: OpponentPolicyConfig) -> Tuple[Optional[Card], Optional[Color]]:
    """
    Choose a *type* of card for the opponent to "play from menu", based on what we
    think exists in hidden_pool, and what matches the current top/active_color.
    Returns (played_card_type, chosen_color_for_wild).
    """
    top = state.top_card
    if top is None:
        return None, None

    # Candidate action types that are legal vs current top and exist in hidden_pool
    candidates: List[Card] = []
    for proto in all_card_types():
        if not proto.matches(top, state.active_color):
            continue
        if hidden_pool_has(state.hidden_pool, proto):
            candidates.append(proto)

    if not candidates:
        return None, None

    # Basic preferences
    def score(proto: Card) -> Tuple[int, int, int]:
        # Higher is better
        is_action = int(proto.rank in (Rank.SKIP, Rank.REVERSE, Rank.DRAW2))
        is_same_color = int((not proto.is_wild()) and (proto.color == state.active_color))
        is_number = int(proto.rank.name.startswith("R"))
        # Order by preference flags
        return (
            is_action if cfg.prefer_actions_first else is_number,
            is_same_color if cfg.prefer_keep_color else 0,
            rng.randint(0, 10_000),  # small tie-breaker
        )

    proto = max(candidates, key=score)

    # For wilds, choose a color—simple rule: keep active color if set, else random
    chosen_color: Optional[Color] = None
    if proto.is_wild():
        chosen_color = state.active_color or rng.choice([Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE])

    return proto, chosen_color







"""
TERMINAL CHECKS / Checking if game is over
"""
def player_total_count(state: GameState, pid: int) -> int:
    p = state.players[pid]
    return len(p.hand) + p.hidden_count

def is_terminal(state: GameState) -> Optional[int]:
    """
    Return the winner's player_id if someone has 0 cards; else None.
    """
    for pid in range(state.num_players()):
        if player_total_count(state, pid) == 0:
            return pid
    return None



"""
Playout after our chosen first move
"""
# ---------- One full playout after our chosen first move ----------

def simulate_to_end(
    state: GameState,
    my_id: int,
    rng: random.Random,
    max_turns: int = 500,  # 300 is enough to finish most games; bump if needed
) -> tuple[bool, Optional[int]]:
    """
    Play out the game using simple baseline policies for *everyone*.
    Returns (finished, winner_id). If finished is False, winner_id is None (timeout).
    """
    turns = 0
    while turns < max_turns:
        turns += 1

        win = is_terminal(state)  # returns winner_id or None
        if win is not None:
            return True, win

        pid = state.current_player
        top = state.top_card
        assert top is not None, "No top card during rollout; state initialization bug?"

        # --- MY TURN: simple baseline: heuristic -> fallback first-legal; draw if none
        if pid == my_id:
            legal = legal_moves_for_player(state, pid)
            if legal:
                next_sz = player_total_count(state, state.next_index(1))
                suggest = (
                    recommend_move(state.players[pid].hand, top, state.active_color, next_sz)
                    or legal[0]
                )
                chosen_color = None
                if suggest.is_wild():
                    chosen_color = my_best_color_from_hand(state.players[pid].hand) \
                                   or state.active_color or Color.RED

                idx = find_hand_index_of_card(state.players[pid].hand, suggest)
                if idx is None:
                    # Fallback if suggest not found (e.g., hand mutated)
                    idx = 0
                    if state.players[pid].hand[idx].is_wild() and chosen_color is None:
                        chosen_color = my_best_color_from_hand(state.players[pid].hand) or Color.RED

                play_card_by_index(state, pid, idx, chosen_color)
            else:
                # Draw 1; if now legal, auto play; else pass
                draw_for_player(state, pid, 1)
                legal2 = legal_moves_for_player(state, pid)
                if legal2:
                    card = legal2[0]
                    idx = find_hand_index_of_card(state.players[pid].hand, card) or 0
                    color = my_best_color_from_hand(state.players[pid].hand) if card.is_wild() else None
                    play_card_by_index(state, pid, idx, color)
                else:
                    pass_turn(state, pid)

        # --- OPPONENT TURN: reasonable baseline policy
        else:
            proto, color = opponent_pick_card_for_menu(state, pid, rng, OpponentPolicyConfig())
            if proto is not None:
                opponent_play_from_menu(state, pid, proto, color)
            else:
                # Draw; auto play if legal, else pass
                draw_for_player(state, pid, 1)
                legal2 = legal_moves_for_player(state, pid)
                if legal2:
                    card = legal2[0]
                    idx = find_hand_index_of_card(state.players[pid].hand, card) or 0
                    color2 = None
                    if card.is_wild():
                        color2 = my_best_color_from_hand(state.players[pid].hand) or state.active_color or Color.RED
                    play_card_by_index(state, pid, idx, color2)
                else:
                    pass_turn(state, pid)

    # Timeout: mark unfinished
    return False, None




"""
Evalutate each of my legal moves with lots of rollouts
"""

@dataclass
class ActionEstimate:
    card: Card                   
    chosen_color: Optional[Color] 
    win_rate: float
    wins: int
    trials: int



def rollout_value_for_action(
    state: GameState,
    my_id: int,
    first_move: Card,
    chosen_color: Optional[Color],
    rng: random.Random,
    n_rollouts: int = 100,
) -> ActionEstimate:
    """
    Evaluate a specific (card type, chosen_color) by averaging win pct over n rollouts.
    """
    wins = 0
    attempts = 0
    for _ in range(n_rollouts):
        s = copy.deepcopy(state)
        idx = find_hand_index_of_card(s.players[my_id].hand, first_move)
        if idx is None:
            continue               # skip, don't count as a trial
        attempts += 1
        play_card_by_index(s, my_id, idx, chosen_color)
        if simulate_to_end(s, my_id, rng) == my_id:
            wins += 1
    trials = max(attempts, 1)
    return ActionEstimate(first_move, chosen_color, wins / trials, wins, trials)




def evaluate_current_position(
    world: GameState,
    my_id_world: int,
    n_rollouts_per_action: int,
    rng_seed: Optional[int] = None,
) -> list[ActionEstimate]:
    """
    Roll out each legal (card, chosen_color) from the given world/seat.
    Guarantees wins <= trials, and never reports wins when trials == 0.
    """
    rng = random.Random(rng_seed)
    legal = legal_moves_for_player(world, my_id_world)
    if not legal:
        return []

    results: list[ActionEstimate] = []

    for card in legal:
        color_choices: list[Optional[Color]] = (
            [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE] if card.is_wild() else [None]
        )

        for chosen_color in color_choices:
            wins = 0
            trials = 0

            for _ in range(n_rollouts_per_action):
                # Clone from the SAME world each time
                s = copy.deepcopy(world)

                # Find a matching card instance in this clone
                idx = find_hand_index_of_card(s.players[my_id_world].hand, card)
                if idx is None:
                    # Rare mismatch from effects; skip this rollout entirely
                    continue

                # Apply candidate action
                play_card_by_index(s, my_id_world, idx, chosen_color)

                # Re-map my seat by name (defensive if list mutated)
                my_id_after = map_pid_by_name(world, s, my_id_world)

                # Simulate to end; normalize return type
                out = simulate_to_end(s, my_id_after, rng, max_turns=300)
                if isinstance(out, tuple) and len(out) == 2:
                    finished, winner = out
                else:
                    finished, winner = True, out

                if not finished:
                    # Don’t count unfinished trajectories
                    continue

                # Count a valid trial
                trials += 1
                if winner == my_id_after:
                    wins += 1

            # Finalize this (card, color) estimate
            win_rate = (wins / trials) if trials else 0.0
            results.append(ActionEstimate(card, chosen_color, win_rate, wins, trials))

    results.sort(key=lambda e: e.win_rate, reverse=True)
    return results



def _action_key(card: Card, chosen_color: Optional[Color]) -> Tuple:
    return (card.rank, card.color, chosen_color)

@dataclass
class ActionAggregate:
    card: Card
    chosen_color: Optional[Color]
    wins: int = 0
    trials: int = 0         # only finished rollouts counted here
    timeouts: int = 0       

    @property
    def win_rate(self) -> float:
        return (self.wins / self.trials) if self.trials > 0 else 0.0


def evaluate_ensemble(
    state: GameState,
    my_id: int,
    n_worlds: int = 8,
    n_rollouts_per_action: int = 32,
    rng_seed: Optional[int] = None,
    force_determinize: bool = False,
) -> List[ActionEstimate]:
    base = random.Random(rng_seed)
    totals: Dict[Tuple, ActionAggregate] = {}

    for _ in range(n_worlds):
        rng_world = random.Random(base.randint(0, 2**31 - 1))

        # Build a simulation world
        if state.manual_mode or force_determinize:
            world = determinize_from_counts(state, rng_world)
        else:
            world = copy.deepcopy(state)

        # Map my seat into this world
        my_id_world = map_pid_by_name(state, world, my_id)

        # Evaluate the position in this world
        ests = evaluate_current_position(
            world,
            my_id_world,
            n_rollouts_per_action,
            rng_seed=rng_world.randint(0, 2**31 - 1),
        )

        # Aggregate results
        for e in ests:
            key = _action_key(e.card, e.chosen_color)
            agg = totals.get(key)
            if agg is None:
                totals[key] = ActionAggregate(e.card, e.chosen_color, e.wins, e.trials)
            else:
                agg.wins += e.wins
                agg.trials += e.trials

    out: List[ActionEstimate] = [
        ActionEstimate(a.card, a.chosen_color, a.win_rate, a.wins, a.trials)
        for a in totals.values()
    ]
    out.sort(key=lambda x: x.win_rate, reverse=True)
    return out


# import copy, random
# from dataclasses import dataclass

# from uknowuno.rules import full_deck

# from typing import List, Tuple, Optional, Dict
# import numpy as np

# from uknowuno.cards import Card, Color, Rank
# from uknowuno.engine import legal_moves_for_player
# from uknowuno.game_state import GameState

# # ----- Action vocabulary (54 actions: 4 colors × (0..9, SKIP, REVERSE, DRAW2) + WILD, WILD_DRAW4)
# COLORS = [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]
# NUMS = [Rank[f"R{i}"] for i in range(10)]
# ACTIONS = [Rank.SKIP, Rank.REVERSE, Rank.DRAW2]
# WILDS = [Rank.WILD, Rank.WILD_DRAW4]

# ACTION_VOCAB: List[Card] = [Card(c, r) for c in COLORS for r in (NUMS + ACTIONS)] + [Card(None, r) for r in WILDS]

# # ----- One-hot helpers
# def one_hot_color(c: Optional[Color]) -> np.ndarray:
#     v = np.zeros(4, dtype=np.float32)
#     if c in COLORS:
#         v[COLORS.index(c)] = 1.0
#     return v

# def one_hot_rank(r: Rank) -> np.ndarray:
#     # order: R0..R9, SKIP, REVERSE, DRAW2, WILD, WILD_DRAW4  => 15 dims
#     order = NUMS + ACTIONS + WILDS
#     v = np.zeros(len(order), dtype=np.float32)
#     v[order.index(r)] = 1.0
#     return v

# def hand_color_counts(hand: List[Card]) -> np.ndarray:
#     v = np.zeros(4, dtype=np.float32)
#     for c in hand:
#         if c.color in COLORS:
#             v[COLORS.index(c.color)] += 1.0
#     return v

# def hand_rank_counts(hand: List[Card]) -> np.ndarray:
#     order = NUMS + ACTIONS + WILDS
#     v = np.zeros(len(order), dtype=np.float32)
#     for c in hand:
#         v[order.index(c.rank)] += 1.0
#     return v

# def others_counts(state: GameState, me: int) -> np.ndarray:
#     # relative order: next, next+1, ... (wrap)
#     out = []
#     for k in range(1, state.num_players()):
#         pid = state.next_index(k) if state.current_player == me else (me + k) % state.num_players()
#         if pid == me: continue
#         p = state.players[pid]
#         out.append(len(p.hand) + p.hidden_count)
#     return np.array(out, dtype=np.float32)

# # ----- Base state features (no action yet)
# def encode_state(state: GameState, me: int) -> np.ndarray:
#     top = state.top_card
#     assert top is not None, "encode_state requires a top card"
#     feats = np.concatenate([
#         one_hot_color(state.active_color),          # 4
#         one_hot_color(top.color),                   # 4
#         one_hot_rank(top.rank),                     # 15
#         hand_color_counts(state.players[me].hand),  # 4
#         hand_rank_counts(state.players[me].hand),   # 15
#         np.array([len(state.players[me].hand)], dtype=np.float32),  # 1
#         others_counts(state, me),                   # N-1
#         np.array([
#             len(state.deck),                        # 1  (0 in manual-mode; okay)
#             len(state.discard),                     # 1
#             1.0 if state.direction == 1 else 0.0    # 1  (clockwise flag)
#         ], dtype=np.float32),
#     ])
#     return feats

# # ----- Action-only features (type + chosen wild color if any)
# def encode_action(card: Card, chosen_color: Optional[Color]) -> np.ndarray:
#     return np.concatenate([
#         one_hot_color(card.color),        # 4 (wilds => zeros)
#         one_hot_rank(card.rank),          # 15
#         one_hot_color(chosen_color),      # 4 (all-zero for non-wilds)
#         np.array([1.0 if card.is_wild() else 0.0], dtype=np.float32),  # 1
#     ])

# def build_examples_for_legal_actions(
#     state: GameState, me: int
# ) -> Tuple[List[np.ndarray], List[Tuple[Card, Optional[Color]]]]:
#     """Return feature vectors and the corresponding (card, chosen_color) for each legal move.
#        For wilds, expand into four color choices."""
#     X: List[np.ndarray] = []
#     acts: List[Tuple[Card, Optional[Color]]] = []
#     base = encode_state(state, me)
#     legal = legal_moves_for_player(state, me)
#     if not legal:
#         return X, acts
#     for card in legal:
#         if card.is_wild():
#             for c in COLORS:
#                 X.append(np.concatenate([base, encode_action(card, c)]))
#                 acts.append((card, c))
#         else:
#             X.append(np.concatenate([base, encode_action(card, None)]))
#             acts.append((card, None))
#     return X, acts

# def feature_dim(state: GameState) -> int:
#     # convenience to see final dimensionality
#     x, _ = build_examples_for_legal_actions(state, state.current_player)
#     return len(x[0]) if x else 0


