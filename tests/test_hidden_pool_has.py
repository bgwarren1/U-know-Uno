from uknowuno.cards import Card, Color, Rank
from ml.rollout_oracle import hidden_pool_has

def test_hidden_pool_has_type():
    pool = [Card.from_text(s) for s in ("R-7", "Y-9", "WILD", "B-REVERSE")]
    assert hidden_pool_has(pool, Card(Color.RED, Rank.R7))
    assert hidden_pool_has(pool, Card(None, Rank.WILD))
    assert not hidden_pool_has(pool, Card(Color.GREEN, Rank.R7))