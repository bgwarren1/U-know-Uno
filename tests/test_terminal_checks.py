from uknowuno.cards import Card, Color, Rank
from uknowuno.engine import start_game_with_my_hand, legal_moves_for_player
from ml.rollout_oracle import player_total_count, is_terminal


def test_is_terminal_detects_winner():
    my_hand = [Card.from_text(s) for s in "R-1, Y-4, G-9, R-3, G-2, B-3, GREEN-REVERSE".split(", ")]
    state = start_game_with_my_hand(
        num_players=4,
        my_hand=my_hand,
        my_index=0,
        initial_top=Card.from_text("R-5"),
        seed=42,
        hand_size=7,
        manual_mode=False,
    )
    # Force Player 1 to be empty to simulate a terminal condition
    p1 = state.players[1]
    p1.hand.clear()
    p1.hidden_count = 0
    assert is_terminal(state) == 1