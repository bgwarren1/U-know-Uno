import random
from uknowuno.cards import Card, Color, Rank
from ml.rollout_oracle import simulate_to_end
from uknowuno.engine import start_game_with_my_hand



def test_simulate_to_end_smoke():
     my_hand = [Card.from_text(s) for s in "R-3, Y-4, G-9, R-3, G-5, B-3, GREEN-REVERSE".split(", ")]
     state = start_game_with_my_hand(
        num_players=4,
        my_hand=my_hand,
        my_index=0,
        initial_top=Card.from_text("R-5"),
        seed=42,
        hand_size=7,
        manual_mode=False,)
     winner = simulate_to_end(state, my_id=0, rng=random.Random(123), max_turns=600)
     assert winner in {0, 1, 2, 3}  
    