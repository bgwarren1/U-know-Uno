from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .cards import Card, Color
from .player import Player

@dataclass
class GameState:
    players: List[Player]
    current_player: int
    direction: int  # +1 clockwise, -1 counterclockwise
    active_color: Color
    deck: List[Card]
    discard: List[Card] = field(default_factory=list)
    # Combined hidden cards that belong to all *opponents* (unknown composition per player)
    hidden_pool: List[Card] = field(default_factory=list)
    # Which seat is you
    my_index: int = 0

    # manual mode: in this mode (realistic online game mode), operator must record every play and top card
    manual_mode: bool = False

    

    
    @property
    def top_card(self) -> Optional[Card]:
        return self.discard[-1] if self.discard else None
        

    def num_players(self) -> int:
        return len(self.players)

    def next_index(self, steps: int = 1) -> int:
        return (self.current_player + steps * self.direction) % self.num_players()

    def advance_turn(self, steps: int = 1) -> None:
        self.current_player = self.next_index(steps)

    def set_active_color(self, c: Color) -> None:
        self.active_color = c

    def summary(self) -> List[Tuple[str, int]]:
        return [(p.name, p.total_count()) for p in self.players]
