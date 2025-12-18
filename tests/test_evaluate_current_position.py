import random
from uknowuno.cards import Card, Color, Rank
from ml.rollout_oracle import evaluate_current_position
from uknowuno.engine import start_game_with_my_hand, legal_moves_for_player



def test_evaluate_current_position_returns_sorted_and_state_unchanged():
    my_hand = [Card.from_text(s) for s in "R-7, Y-4, G-9, R-3, G-2, B-3, B-9".split(", ")]
    state = start_game_with_my_hand(
        num_players=4,
        my_hand=my_hand,
        my_index=0,
        initial_top=Card.from_text("R-5"),
        seed=42,
        hand_size=7,
        manual_mode=False,
    )
    before = (list(state.discard), state.current_player)
    estimates = evaluate_current_position(state, my_id=0, n_rollouts_per_action=8, rng_seed=7)
    # Should have at least one estimate (one per legal move; wilds collapsed to best color)
    assert len(estimates) >= 1
    # Sorted descending by win_rate
    rates = [e.win_rate for e in estimates]
    assert rates == sorted(rates, reverse=True)
    
    after = (list(state.discard), state.current_player)
    assert before == after