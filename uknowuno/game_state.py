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

    # FIX (opponent-hands): rollout-only flag. True once a world has been "dealt out" so that
    # every player -- including opponents -- holds their OWN individual hand (Player.hand) instead
    # of sharing one combined hidden_pool. Simulations use this to make each opponent play from
    # their own cards, and to route drawn cards into the correct player's hand.
    all_hands_known: bool = False

    

    
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
