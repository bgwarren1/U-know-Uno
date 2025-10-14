import random
from uknowuno.cards import Card, Color, Rank
from ml.rollout_oracle import rollout_value_for_action
from uknowuno.engine import start_game_with_my_hand, legal_moves_for_player

def test_rollout_value_for_action_bounds_and_trials():
    my_hand = [Card.from_text(s) for s in "R-3, Y-4, G-9, R-3, G-5, B-3, GREEN-REVERSE".split(", ")]
    state = start_game_with_my_hand(
        num_players=4,
        my_hand=my_hand,
        my_index=0,
        initial_top=Card.from_text("R-5"),
        seed=42,
        hand_size=7,
        manual_mode=False,)
    legal = legal_moves_for_player(state, 0)
    assert len(legal) > 0
    first = legal[0]
    est = rollout_value_for_action(state, my_id=0, first_move=first, chosen_color=None, rng=random.Random(123), n_rollouts=8)
    assert est.trials == 8
    assert 0.0 <= est.win_rate <= 1.0