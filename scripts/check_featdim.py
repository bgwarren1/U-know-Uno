from ml.featurize import feature_dim
from uknowuno.engine import start_game_with_my_hand
from uknowuno.cards import Card, Color, Rank

def dummy_hand(k=7):
    # 7 RED cards including R1 so it matches the top below
    pool = [Rank.R1, Rank.R2, Rank.R3, Rank.R4, Rank.R6, Rank.R7, Rank.R8, Rank.R9]
    return [Card(Color.RED, r) for r in pool[:k]]

def fresh_game(n_players: int, seed: int = 1):
    top = Card(Color.RED, Rank.R1)  # matches dummy_hand
    g = start_game_with_my_hand(
        num_players=n_players,
        my_hand=dummy_hand(7),
        my_index=0,
        seed=seed,
        hand_size=7,
        initial_top=top,
        initial_active_color=top.color,
        manual_mode=True
    )
    return g

for n in (2, 3, 4):
    g = fresh_game(n)
    print(n, feature_dim(g))

