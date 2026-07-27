"""Regression guard for the opponent-hands fix.

Background: simulations used to pour every opponent's cards into ONE shared `hidden_pool`,
so each opponent effectively played from the whole combined pile (14-21 cards) instead of
their own 7. That drove my seat (seat 0) down to ~4% wins in 4p (fair share = 25%) and left
some seats unable to win at all. The fix deals each opponent an individual hand during
determinization. These tests lock that in: hands are dealt out, cards are conserved, and the
simulator is roughly fair with no seat starved.
"""
import collections
import random

from uknowuno.cards import Card, Color, Rank
from uknowuno.engine import start_game_with_my_hand
from uknowuno.rules import full_deck
from ml.rollout_oracle import determinize_from_counts, simulate_to_end


def _rand_hand(rng):
    """A random 7-card hand that can't collide with the R-5 starting top."""
    deck = [c for c in full_deck() if not c.is_wild() and c.rank != Rank.R5]
    rng.shuffle(deck)
    return deck[:7]


def _make_manual_start(n_players, rng):
    return start_game_with_my_hand(
        num_players=n_players,
        my_hand=_rand_hand(rng),
        my_index=0,
        initial_top=Card(Color.RED, Rank.R5),
        initial_active_color=Color.RED,
        hand_size=7,
        manual_mode=True,
    )


def test_determinize_deals_individual_hands_and_conserves_cards():
    state = _make_manual_start(4, random.Random(0))
    world = determinize_from_counts(state, random.Random(1))

    # Every player now holds their own real hand; the shared pool is emptied.
    assert world.all_hands_known is True
    assert world.hidden_pool == []
    for p in world.players:
        assert len(p.hand) == 7
        assert p.hidden_count == 0

    # All 108 cards are accounted for exactly once (hands + deck + discard), no leaks/dupes.
    everything = [c for p in world.players for c in p.hand] + world.deck + world.discard
    assert len(everything) == len(full_deck()) == 108


def _win_shares(n_players, n_games, seed):
    rng = random.Random(seed)
    wins = collections.Counter()
    finished = 0
    for _ in range(n_games):
        state = _make_manual_start(n_players, rng)
        world = determinize_from_counts(state, random.Random(rng.randint(0, 10**9)))
        done, winner = simulate_to_end(world, 0, rng, max_turns=800)
        if done:
            finished += 1
            wins[winner] += 1
    return wins, finished


def test_simulation_is_roughly_fair_no_seat_starved():
    # Deterministic (fixed seeds), so the bounds below are stable, not flaky.
    for n_players in (2, 3, 4):
        wins, finished = _win_shares(n_players, n_games=120, seed=0)
        assert finished == 120  # every game reaches a winner
        fair = 1 / n_players
        shares = {pid: wins[pid] / finished for pid in range(n_players)}

        # No seat is starved. The bug starved seat 0 (~0.04) and sometimes others (0.00).
        for pid in range(n_players):
            assert shares[pid] > 0.5 * fair, (n_players, shares)

        # My own seat lands in a sane band around fair -- never near the broken collapse.
        assert 0.5 * fair <= shares[0] <= 1.6 * fair, (n_players, shares)
