import random
from uknowuno.cards import Card, Color, Rank
from uknowuno.engine import start_game_with_my_hand
from ml.rollout_oracle import opponent_pick_card_for_menu, OpponentPolicyConfig

def test_opponent_pick_card_for_menu_basic_choice():
    # Start a deterministic game (normal mode so there’s a real deck)
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

    # Make top RED and ensure hidden_pool contains a RED match
    state.discard[-1] = Card(Color.RED, Rank.R5)
    state.set_active_color(Color.RED)
    state.hidden_pool = [Card(Color.RED, Rank.R7)]
    state.players[1].hidden_count = 1  

    proto, chosen_color = opponent_pick_card_for_menu(
        state, pid=1, rng=random.Random(0), cfg=OpponentPolicyConfig()
    )

    assert proto is not None
    assert proto.matches(state.top_card, state.active_color)
