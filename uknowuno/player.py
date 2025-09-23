from dataclasses import dataclass, field
from typing import List
from .cards import Card

@dataclass
class Player:
    id: int
    name: str
    # Known, visible cards (only your seat will use this; opponents’ hands are unknown)
    hand: List[Card] = field(default_factory=list)
    # Hidden/unknown cards count (opponents)
    hidden_count: int = 0

    # flags
    said_uno: bool = False
    is_human: bool = True

    def visible_count(self) -> int:
        return len(self.hand)

    def total_count(self) -> int:
        return self.visible_count() + self.hidden_count
