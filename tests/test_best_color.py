from uknowuno.cards import Card, Color, Rank
from ml.rollout_oracle import my_best_color_from_hand

def test_my_best_color_from_hand_picks_most_frequent():
    hand = [Card.from_text(s) for s in ("R-1", "R-2", "B-3", "G-4", "WILD")]
    assert my_best_color_from_hand(hand) == Color.RED
    assert my_best_color_from_hand([Card.from_text("WILD")]) is None