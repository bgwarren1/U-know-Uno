from uknowuno.cards import Card

def test_parse_my_hand():
    txt = "R-7, Y-4, G-9, R-5, G-2, B-3, B-9"
    cards = [Card.from_text(p.strip()) for p in txt.split(",")]
    assert len(cards) == 7
    assert all(isinstance(c, Card) for c in cards)
