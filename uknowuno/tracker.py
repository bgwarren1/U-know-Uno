from collections import Counter
from typing import List, Dict
from .cards import Card

"""
NEED TO REVIEW THIS, CARDS ARE NOT BEING REMOVED FROM CARD SELECTION DROPDOWN
"""


class PlayTracker:
    """
    Tracks what has been seen to help drive simple heuristics now
    and probability later.
    """
    def __init__(self):
        self.played: List[Card] = []
        self.counts = Counter()

    def record(self, card: Card):
        self.played.append(card)
        self.counts[card.rank] += 1

    def times_number_seen(self, number_str: str) -> int:
        key = f"R{number_str}"
        return sum(v for k, v in self.counts.items() if k.value == key)
