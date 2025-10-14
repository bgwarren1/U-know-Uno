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

def simulate_to_end(state: GameState, my_id: int, rng: random.Random, max_turns: int = 600) -> int:
    """
    Run a complete game using simple baseline policies for *everyone*.
    Returns the winner's player_id.
    """
    turns = 0
    while turns < max_turns:
        turns += 1
        win = is_terminal(state)
        if win is not None:
            return win

        pid = state.current_player
        top = state.top_card
        assert top is not None, "No top card during rollout; state initialization bug?"

        # My turn: use strategy.recommend_move as a baseline
        if pid == my_id:
            legal = legal_moves_for_player(state, pid)
            if legal:
                # Suggest a card from the heuristic; fallback to first legal if None
                next_player_size = player_total_count(state, state.next_index(1))
                suggest = recommend_move(state.players[pid].hand, top, state.active_color, next_player_size) or legal[0]
                chosen_color = None
                if suggest.is_wild():
                    chosen_color = my_best_color_from_hand(state.players[pid].hand) or state.active_color or Color.RED
                idx = find_hand_index_of_card(state.players[pid].hand, suggest)
                if idx is None:
                    
                    idx = 0
                    chosen_color = chosen_color or (my_best_color_from_hand(state.players[pid].hand) or Color.RED)
                play_card_by_index(state, pid, idx, chosen_color)
            else:
                # Draw one; if now legal, play it; else pass
                draw_for_player(state, pid, 1)
                legal2 = legal_moves_for_player(state, pid)
                if legal2:
                    card = legal2[0]
                    idx = find_hand_index_of_card(state.players[pid].hand, card) or 0
                    color = my_best_color_from_hand(state.players[pid].hand) if card.is_wild() else None
                    play_card_by_index(state, pid, idx, color)
                else:
                    pass_turn(state, pid)

        # Opponent turn: choose a type that exists in hidden_pool and is legal; else draw, then pass
        else:
            proto, color = opponent_pick_card_for_menu(state, pid, rng, OpponentPolicyConfig())
            if proto is not None:
                opponent_play_from_menu(state, pid, proto, color)
            else:
                draw_for_player(state, pid, 1)
                pass_turn(state, pid)

    # Safety stop: if we somehow loop too long, pick winner by fewest cards (rare)
    sizes = [(player_total_count(state, pid), pid) for pid in range(state.num_players())]
    sizes.sort()
    return sizes[0][1]




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
    for _ in range(n_rollouts):
        # Deep copy the whole game state since each trial gotta start from same spot
        s = copy.deepcopy(state)

        # Play the chosen move on the copy
        idx = find_hand_index_of_card(s.players[my_id].hand, first_move)
        if idx is None:
            continue
        play_card_by_index(s, my_id, idx, chosen_color)

        winner = simulate_to_end(s, my_id, rng)
        if winner == my_id:
            wins += 1

    trials = n_rollouts
    return ActionEstimate(first_move, chosen_color, wins / trials if trials > 0 else 0.0, wins, trials)




def evaluate_current_position(
    state: GameState,
    my_id: int,
    n_rollouts_per_action: int = 64,
    rng_seed: Optional[int] = None,
) -> List[ActionEstimate]:
    """
    For the *current* state and the current player == my_id,
    return a list of ActionEstimate, one per legal move (wilds expanded over colors).
    """
    assert state.current_player == my_id, "This evaluator expects it's your turn."

    rng = random.Random(rng_seed)

    legal = legal_moves_for_player(state, my_id)
    estimates: List[ActionEstimate] = []

    if not legal:
        # No moves
        return estimates

    # For each legal move: if wild, evaluate all 4 color choices and keep the best;
    # otherwise evaluate the card as-is.
    for card in legal:
        if card.is_wild():
            colors = [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]
            best: Optional[ActionEstimate] = None
            for c in colors:
                est = rollout_value_for_action(state, my_id, card, c, rng, n_rollouts_per_action)
                if (best is None) or (est.win_rate > best.win_rate):
                    best = est
            if best:
                estimates.append(best)
        else:
            est = rollout_value_for_action(state, my_id, card, None, rng, n_rollouts_per_action)
            estimates.append(est)

    # Sort best-to-worst by win rate
    estimates.sort(key=lambda e: e.win_rate, reverse=True)
    return estimates



