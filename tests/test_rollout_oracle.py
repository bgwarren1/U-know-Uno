# tests/test_rollout_oracle.py
import random

from uknowuno.cards import Card, Color, Rank
from uknowuno.engine import start_game_with_my_hand, legal_moves_for_player
from ml.rollout_oracle import (
    all_card_types,
    find_hand_index_of_card,
    my_best_color_from_hand,
    hidden_pool_has,
    opponent_pick_card_for_menu,
    is_terminal,
    simulate_to_end,
    rollout_value_for_action,
    evaluate_current_position,
)

# ---------- Fixtures (inline to keep it simple) ----------

def make_start_state():
    """
    Deterministic 4-player game, your hand fixed, starting top = R-5.
    Normal (non-manual) mode so rollouts can draw from the deck.
    """
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
    return state


# ---------- Unit tests for small helpers ----------

def test_all_card_types_has_54_and_contains_examples():
    types = all_card_types()
    assert len(types) == 54
    assert Card(Color.BLUE, Rank.R7) in types
    assert Card(Color.GREEN, Rank.DRAW2) in types
    assert Card(None, Rank.WILD) in types
    assert Card(None, Rank.WILD_DRAW4) in types

def test_find_hand_index_of_card_matches_by_type():
    hand = [Card.from_text(s) for s in ("R-7", "B-7", "WILD", "G-2")]
    # exact color+rank for non-wild
    assert find_hand_index_of_card(hand, Card(Color.RED, Rank.R7)) == 0
    # wilds match by rank only (color=None in hand)
    assert find_hand_index_of_card(hand, Card(None, Rank.WILD)) == 2
    # not present
    assert find_hand_index_of_card(hand, Card(Color.YELLOW, Rank.R9)) is None

def test_my_best_color_from_hand_picks_most_frequent():
    hand = [Card.from_text(s) for s in ("R-1", "R-2", "B-3", "G-4", "WILD")]
    assert my_best_color_from_hand(hand) == Color.RED
    assert my_best_color_from_hand([Card.from_text("WILD")]) is None

def test_hidden_pool_has_type():
    pool = [Card.from_text(s) for s in ("R-7", "Y-9", "WILD", "B-REVERSE")]
    assert hidden_pool_has(pool, Card(Color.RED, Rank.R7))
    assert hidden_pool_has(pool, Card(None, Rank.WILD))
    assert not hidden_pool_has(pool, Card(Color.GREEN, Rank.R7))

# ---------- Tests that touch GameState logic (no training; tiny sims) ----------

def test_is_terminal_detects_winner():
    state = make_start_state()
    # Force Player 1 to be empty to simulate a terminal condition
    p1 = state.players[1]
    p1.hand.clear()
    p1.hidden_count = 0
    assert is_terminal(state) == 1

def test_opponent_pick_card_for_menu_basic_choice():
    state = make_start_state()
    # Make top require RED or rank match; ensure hidden_pool has a matching RED card
    state.discard[-1] = Card(Color.RED, Rank.R5)  # top card
    state.set_active_color(Color.RED)
    # Ensure hidden_pool contains a RED-7
    state.hidden_pool = [Card(Color.RED, Rank.R7)]
    for pid in range(state.num_players()):
        if pid != state.my_index:
            state.players[pid].hidden_count = 1
            break
    proto, chosen_color = opponent_pick_card_for_menu(state, pid=1, rng=random.Random(0), cfg=None)  # cfg default in function
    assert proto is not None
    assert proto.color == Color.RED or proto.is_wild()
    # Wilds allowed, chosen_color can be None or a Color; we only assert type correctness
    if proto.is_wild():
        assert (chosen_color is None) or isinstance(chosen_color, Color)

def test_simulate_to_end_smoke():
    state = make_start_state()
    winner = simulate_to_end(state, my_id=0, rng=random.Random(123), max_turns=600)
    assert winner in {0, 1, 2, 3}

def test_rollout_value_for_action_bounds_and_trials():
    state = make_start_state()
    legal = legal_moves_for_player(state, 0)
    assert len(legal) > 0
    first = legal[0]
    est = rollout_value_for_action(state, my_id=0, first_move=first, chosen_color=None, rng=random.Random(123), n_rollouts=8)
    assert est.trials == 8
    assert 0.0 <= est.win_rate <= 1.0

def test_evaluate_current_position_returns_sorted_and_state_unchanged():
    state = make_start_state()
    before = (list(state.discard), state.current_player)
    estimates = evaluate_current_position(state, my_id=0, n_rollouts_per_action=8, rng_seed=7)
    # Should have at least one estimate (one per legal move; wilds collapsed to best color)
    assert len(estimates) >= 1
    # Sorted descending by win_rate
    rates = [e.win_rate for e in estimates]
    assert rates == sorted(rates, reverse=True)
    # Original state should be unchanged (we deep-copy inside)
    after = (list(state.discard), state.current_player)
    assert before == after
