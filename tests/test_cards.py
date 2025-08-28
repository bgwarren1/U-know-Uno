from uknowuno.cards import Card, Color, Rank

def test_basic_match():
    top = Card(Color.RED, Rank.R5)
    c1 = Card(Color.RED, Rank.R2)
    c2 = Card(Color.GREEN, Rank.R5)
    c3 = Card(None, Rank.WILD)
    assert c1.matches(top, Color.RED)
    assert c2.matches(top, Color.RED)
    assert c3.matches(top, Color.RED)
