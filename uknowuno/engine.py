from __future__ import annotations
import random
from typing import List, Optional, Tuple

from .cards import Card, Color, Rank
from .rules import full_deck, legal_moves
from .player import Player
from .game_state import GameState

# ---------- Utility ----------

def new_shuffled_deck(seed: Optional[int] = None) -> List[Card]:
    deck = full_deck()
    rng = random.Random(seed)
    rng.shuffle(deck)
    return deck

def _refill_deck_from_discard(state: GameState):
    """If deck is empty, shuffle all but top discard back into deck."""
    if len(state.deck) > 0:
        return
    if len(state.discard) <= 1:
        return
    top = state.discard[-1]
    pool = state.discard[:-1][:]
    random.shuffle(pool)
    state.deck = pool
    state.discard = [top]

def _draw_from_deck(state: GameState, n: int) -> List[Card]:
    drawn: List[Card] = []
    for _ in range(n):
        if not state.deck:
            _refill_deck_from_discard(state)
            if not state.deck:
                break
        drawn.append(state.deck.pop())
    return drawn

def _remove_one_card_from_list(pool: List[Card], target: Card) -> bool:
    """Remove one card equal to target (match by rank and color, wilds by rank)."""
    for i, c in enumerate(pool):
        if c.rank == target.rank:
            # for wilds, color is None in deck/pool; for colored, require exact color
            if c.is_wild() or c.color == target.color:
                pool.pop(i)
                return True
    return False

def pick_best_color(hand: List[Card]) -> Optional[Color]:
    colors = [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]
    counts = {c: 0 for c in colors}
    for h in hand:
        if h.color in counts:
            counts[h.color] += 1  # type: ignore
    best = max(counts.items(), key=lambda kv: (kv[1], -colors.index(kv[0])))
    return best[0] if best[1] > 0 else None

# ---------- Start a game with YOUR known hand ----------

def start_game_with_my_hand(
    num_players: int,
    my_hand: List[Card],
    my_index: int = 0,
    names: Optional[List[str]] = None,
    seed: Optional[int] = None,
    hand_size: int = 7,
    initial_top: Optional[Card] = None,
    initial_active_color: Optional[Color] = None,
) -> GameState:
    assert 2 <= num_players <= 10, "Uno typically 2–10 players."
    assert len(my_hand) == hand_size, f"Expected {hand_size} cards for your hand."

    deck = new_shuffled_deck(seed)

    # Guard: starting top can't be one of your 7 (that would duplicate it)
    if initial_top is not None and any(
        (c.rank == initial_top.rank) and (c.color == initial_top.color)
        for c in my_hand
    ):
        raise ValueError("Starting top card conflicts with your hand; pick a different top or change your hand.")

    # Remove your exact cards from the deck
    for card in my_hand:
        ok = _remove_one_card_from_list(deck, card)
        if not ok:
            raise ValueError(f"Your card {card.short()} not available in deck (duplicate/typo?)")

    # Build players
    if not names:
        names = [f"Player {i}" for i in range(num_players)]
    players = [Player(id=i, name=names[i]) for i in range(num_players)]
    players[my_index].hand = my_hand[:]
    players[my_index].hidden_count = 0

    # Deal opponents into hidden_pool
    hidden_pool: List[Card] = []
    for pid in range(num_players):
        if pid == my_index:
            continue
        # draw hand_size from deck
        drawn = []
        for _ in range(hand_size):
            if not deck:
                break
            drawn.append(deck.pop())
        hidden_pool.extend(drawn)
        players[pid].hidden_count = len(drawn)

    discard: List[Card] = []
    active_color: Color = Color.RED  # default fallback

    # If user provided a specific starting top, use it (and keep counts consistent)
    if initial_top is not None:
        removed = _remove_one_card_from_list(deck, initial_top)
        if not removed:
            # Try removing it from hidden_pool, and decrement any opponent's hidden_count
            removed = _remove_one_card_from_list(hidden_pool, initial_top)
            if removed:
                for pid in range(num_players):
                    if pid != my_index and players[pid].hidden_count > 0:
                        players[pid].hidden_count -= 1
                        break
        if not removed:
            raise ValueError(f"Starting top card {initial_top.short()} not available in shoe (deck/hidden pool).")

        discard = [initial_top]
        if initial_top.is_wild():
            # Use provided active color or default RED
            active_color = initial_active_color or Color.RED
        else:
            active_color = initial_top.color  # type: ignore[assignment]

        return GameState(
            players=players,
            current_player=0,
            direction=1,
            active_color=active_color,
            deck=deck,
            discard=discard,
            hidden_pool=hidden_pool,
            my_index=my_index,
        )

    # Otherwise: flip a reasonable top from the deck (prefer colored number)
    while deck:
        top = deck.pop()
        if (not top.is_wild()) and top.rank.name.startswith("R"):
            discard.append(top)
            active_color = top.color  # type: ignore[assignment]
            break
        else:
            # Put it to bottom and keep trying a bit; if deck gets small, accept it
            deck.insert(0, top)
            if len(discard) == 0 and len(deck) < 20:
                discard.append(top)
                active_color = (top.color if not top.is_wild() else Color.RED)
                break

    return GameState(
        players=players,
        current_player=0,
        direction=1,
        active_color=active_color,
        deck=deck,
        discard=discard,
        hidden_pool=hidden_pool,
        my_index=my_index,
    )


# ---------- Legality helper for YOUR visible hand ----------

def legal_moves_for_player(state: GameState, player_id: int) -> List[Card]:
    hand = state.players[player_id].hand
    top = state.top_card
    if top is None:
        return []
    return legal_moves(hand, top, state.active_color)

# ---------- Your play (by visible card index) ----------

def play_card_by_index(
    state: GameState,
    player_id: int,
    hand_idx: int,
    chosen_color: Optional[Color] = None,
) -> Tuple[bool, str]:
    if player_id != state.current_player:
        return False, f"Not {state.players[player_id].name}'s turn."

    if hand_idx < 0 or hand_idx >= len(state.players[player_id].hand):
        return False, "Invalid card index."

    top = state.top_card
    if top is None:
        return False, "No top card in discard."

    card = state.players[player_id].hand[hand_idx]
    if card not in legal_moves_for_player(state, player_id):
        return False, f"{card.short()} is not a legal move."

    # Move card to discard
    state.players[player_id].hand.pop(hand_idx)
    state.discard.append(card)

    # Active color
    if card.is_wild():
        if chosen_color is None:
            chosen_color = pick_best_color(state.players[player_id].hand) or state.active_color
        state.set_active_color(chosen_color)
    else:
        state.set_active_color(card.color)  # type: ignore

    return _apply_action_and_advance(state, card)

# ---------- Opponent play from MENU (unknown hands) ----------

def opponent_play_from_menu(
    state: GameState,
    player_id: int,
    played_card: Card,
    chosen_color: Optional[Color] = None,
) -> Tuple[bool, str]:
    if player_id == state.my_index:
        return False, "Use your own Play button for your seat."
    if player_id != state.current_player:
        return False, f"Not {state.players[player_id].name}'s turn."
    if state.top_card is None:
        return False, "No top card."

    # Must be legal w.r.t. top + active color
    if not played_card.matches(state.top_card, state.active_color):
        return False, f"{played_card.short()} doesn't match top/active color."

    # Reduce their hidden count
    if state.players[player_id].hidden_count <= 0:
        return False, "That player has no cards to play."
    state.players[player_id].hidden_count -= 1

    # Remove such a card from hidden_pool if possible; otherwise try from deck; else just accept (best-effort)
    if not _remove_one_card_from_list(state.hidden_pool, played_card):
        _remove_one_card_from_list(state.deck, played_card)

    # Place on discard and set color
    state.discard.append(played_card)
    if played_card.is_wild():
        if chosen_color is None:
            chosen_color = state.active_color
        state.set_active_color(chosen_color)
    else:
        state.set_active_color(played_card.color)  # type: ignore

    return _apply_action_and_advance(state, played_card)

# ---------- Draw / Pass helpers ----------

def draw_for_player(state: GameState, player_id: int, n: int = 1) -> Tuple[bool, str, List[Card]]:
    if player_id != state.current_player:
        return False, f"Not {state.players[player_id].name}'s turn.", []
    drawn = _draw_from_deck(state, n)
    if player_id == state.my_index:
        state.players[player_id].hand.extend(drawn)
    else:
        state.hidden_pool.extend(drawn)
        state.players[player_id].hidden_count += len(drawn)
    return True, f"Drew {len(drawn)}", drawn

def pass_turn(state: GameState, player_id: int) -> Tuple[bool, str]:
    if player_id != state.current_player:
        return False, f"Not {state.players[player_id].name}'s turn."
    state.advance_turn(1)
    return True, "Passed."

# ---------- Apply action effects & advance ----------

def _apply_action_and_advance(state: GameState, card: Card) -> Tuple[bool, str]:
    msg = ""
    np_index = state.next_index(1)

    if card.rank == Rank.REVERSE:
        if state.num_players() == 2:
            state.advance_turn(2)
            return True, "Reverse (acts like Skip in 2 players)."
        state.direction *= -1
        state.advance_turn(1)
        return True, "Reverse: direction changed."

    if card.rank == Rank.SKIP:
        state.advance_turn(2)
        return True, "Skip: next player skipped."

    if card.rank == Rank.DRAW2:
        # Next player draws 2 and loses turn
        drawn = _draw_from_deck(state, 2)
        if np_index == state.my_index:
            state.players[np_index].hand.extend(drawn)
        else:
            state.hidden_pool.extend(drawn)
            state.players[np_index].hidden_count += len(drawn)
        state.advance_turn(2)
        return True, f"{state.players[np_index].name} draws 2 and is skipped."

    if card.rank == Rank.WILD_DRAW4:
        drawn = _draw_from_deck(state, 4)
        if np_index == state.my_index:
            state.players[np_index].hand.extend(drawn)
        else:
            state.hidden_pool.extend(drawn)
            state.players[np_index].hidden_count += len(drawn)
        state.advance_turn(2)
        return True, f"{state.players[np_index].name} draws 4 and is skipped."

    # Normal / WILD
    state.advance_turn(1)
    return True, "Played."
