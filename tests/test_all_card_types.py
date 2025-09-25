from uknowuno.cards import Card, Color, Rank
from ml.rollout_oracle import all_card_types

def test_all_card_types_has_54_and_contains_examples():
    types = all_card_types()
    assert len(types) == 54
    assert Card(Color.BLUE, Rank.R7) in types
    assert Card(Color.GREEN, Rank.DRAW2) in types
    assert Card(None, Rank.WILD) in types
    assert Card(None, Rank.WILD_DRAW4) in types
