from typing import List, Optional
from .cards import Card, Color
from .strategy import recommend_move


def recommend_from_text(hand_txt: List[str], top_txt: str, active_color: str, next_n: Optional[int] = None) -> Optional[Card]:
    hand = [Card.from_text(t) for t in hand_txt]
    top = Card.from_text(top_txt)
    color = Color[active_color] if len(active_color) > 1 else Color(active_color)
    return recommend_move(hand, top, color, next_player_hand_size=next_n)
