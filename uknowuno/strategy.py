from typing import List, Optional
from .cards import Card, Color, Rank
from .rules import legal_moves

# Will replace, might want to keep some of the heuristic though after training
def recommend_move(
    hand: List[Card],
    top: Card,
    active_color: Color,
    next_player_hand_size: Optional[int] = None,
) -> Optional[Card]:
    """
    Very simple first-pass heuristic:
    1) If next player is low on cards (<=2), prefer SKIP/DRAW2/REVERSE if legal.
    2) Prefer playing non-wilds to save wilds.
    3) Among legal non-wilds, prefer the color you have most of in your remaining hand.
    4) If none, use WILD to set to your best color; if only WILD_DRAW4, do that.
    """
    moves = legal_moves(hand, top, active_color)
    if not moves:
        return None

    # 1) punish next player if possible
    if next_player_hand_size is not None and next_player_hand_size <= 2:
        for prio in [Rank.DRAW2, Rank.SKIP, Rank.REVERSE]:
            for m in moves:
                if m.rank == prio:
                    return m

    non_wilds = [m for m in moves if not m.is_wild()]
    if non_wilds:
        # 3) prefer color we own most of
        color_counts = {}
        for c in [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]:
            color_counts[c] = sum(1 for h in hand if (h.color == c))
        non_wilds.sort(key=lambda c: color_counts.get(c.color, 0), reverse=True)  # type: ignore[index]
        return non_wilds[0]

    # 4) Wild fallback
    for m in moves:
        if m.rank in (Rank.WILD, Rank.WILD_DRAW4):
            return m

    return moves[0]
