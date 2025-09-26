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


