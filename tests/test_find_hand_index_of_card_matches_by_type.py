from uknowuno.cards import Card, Color, Rank
from ml.rollout_oracle import find_hand_index_of_card





def test_find_hand_index_of_card_matches_by_type():
    hand = [Card.from_text(s) for s in ("R-7", "B-7", "WILD", "G-2")]
    # exact color+rank for non-wild
    assert find_hand_index_of_card(hand, Card(Color.RED, Rank.R7)) == 0
    # wilds match by rank only (color=None in hand)
    assert find_hand_index_of_card(hand, Card(None, Rank.WILD)) == 2
    # not present
    assert find_hand_index_of_card(hand, Card(Color.YELLOW, Rank.R9)) is None