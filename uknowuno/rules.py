from typing import List
from .cards import Card, Color, Rank


def full_deck() -> List[Card]:
    """
    Standard 108-card UNO deck (classic rules, no 7-0).
    - For each color (R,Y,G,B):
      1x 0, 2x each of 1..9, 2x SKIP, 2x REVERSE, 2x DRAW2
    - 4x WILD, 4x WILD_DRAW4
    """
    deck: List[Card] = []
    colors = [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]

    for c in colors:
        deck.append(Card(c, Rank.R0))
        for r in [Rank.R1, Rank.R2, Rank.R3, Rank.R4, Rank.R5, Rank.R6, Rank.R7, Rank.R8, Rank.R9]:
            deck.extend([Card(c, r), Card(c, r)])
        for a in [Rank.SKIP, Rank.REVERSE, Rank.DRAW2]:
            deck.extend([Card(c, a), Card(c, a)])

    deck.extend([Card(None, Rank.WILD) for _ in range(4)])
    deck.extend([Card(None, Rank.WILD_DRAW4) for _ in range(4)])
    return deck


def legal_moves(hand: List[Card], top: Card, active_color: Color) -> List[Card]:
    return [c for c in hand if c.matches(top, active_color)]
