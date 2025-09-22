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
    manual_mode: bool = False,           # NEW
) -> GameState:
    assert 2 <= num_players <= 10
    assert len(my_hand) == hand_size

    # Build players
    if not names:
        names = [f"Player {i}" for i in range(num_players)]
    players = [Player(id=i, name=names[i]) for i in range(num_players)]
    players[my_index].hand = my_hand[:]
    players[my_index].hidden_count = 0

    # MANUAL MODE: no deck, no hidden pool; opponents just start with counts
    if manual_mode:
        # Opponents start with 7 hidden cards (counts only)
        for pid in range(num_players):
            if pid != my_index:
                players[pid].hidden_count = hand_size
        discard: List[Card] = []
        if initial_top is None:
            raise ValueError("Manual mode requires an explicit starting top card.")
        discard = [initial_top]
        active_color = (
            initial_active_color
            or (initial_top.color if not initial_top.is_wild() else Color.RED)
            or Color.RED
        )
        return GameState(
            players=players,
            current_player=0,
            direction=1,
            active_color=active_color,
            deck=[],                 # not used
            discard=discard,
            hidden_pool=[],          # not used
            my_index=my_index,
            manual_mode=True,
        )

    # NORMAL MODE (non-manual): keep your previous behavior (deck + hidden_pool)
    deck = new_shuffled_deck(seed)

    # safeguard: chosen top cannot also be in your hand (would duplicate)
    if initial_top is not None and any(
        (c.rank == initial_top.rank) and (c.color == initial_top.color)
        for c in my_hand
    ):
        raise ValueError("Starting top card conflicts with your hand.")

    # remove your exact hand from deck
    for card in my_hand:
        if not _remove_one_card_from_list(deck, card):
            raise ValueError(f"Your card {card.short()} not available in deck.")

    # deal opponents into hidden_pool
    hidden_pool: List[Card] = []
    for pid in range(num_players):
        if pid == my_index:
            continue
        drawn = []
        for _ in range(hand_size):
            if not deck:
                break
            drawn.append(deck.pop())
        hidden_pool.extend(drawn)
        players[pid].hidden_count = len(drawn)

    discard: List[Card] = []
    active_color: Color = Color.RED

    if initial_top is not None:
        removed = _remove_one_card_from_list(deck, initial_top)
        if not removed:
            removed = _remove_one_card_from_list(hidden_pool, initial_top)
            if removed:
                for pid in range(num_players):
                    if pid != my_index and players[pid].hidden_count > 0:
                        players[pid].hidden_count -= 1
                        break
        if not removed:
            raise ValueError(f"Starting top {initial_top.short()} not available.")
        discard = [initial_top]
        active_color = (
            initial_active_color
            or (initial_top.color if not initial_top.is_wild() else Color.RED)
            or Color.RED
        )
    else:
        # flip a reasonable top
        while deck:
            top = deck.pop()
            if (not top.is_wild()) and top.rank.name.startswith("R"):
                discard.append(top)
                active_color = top.color  # type: ignore
                break
            else:
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

    if not played_card.matches(state.top_card, state.active_color):
        return False, f"{played_card.short()} doesn't match top/active color."

    # decrement their count
    if state.players[player_id].hidden_count <= 0:
        return False, "That player has no cards to play."
    state.players[player_id].hidden_count -= 1

    # In manual mode we don't maintain hidden_pool/deck identity
    if not state.manual_mode:
        # best effort to keep pools consistent
        from_pool = _remove_one_card_from_list(state.hidden_pool, played_card)
        if not from_pool:
            _remove_one_card_from_list(state.deck, played_card)

    # discard & color
    state.discard.append(played_card)
    if played_card.is_wild():
        state.set_active_color(chosen_color or state.active_color)
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
    np_index = state.next_index(1)

    # REVERSE
    if card.rank == Rank.REVERSE:
        if state.num_players() == 2:
            state.advance_turn(2)
            return True, "Reverse (2p => Skip)."
        state.direction *= -1
        state.advance_turn(1)
        return True, "Reverse: direction changed."

    # SKIP
    if card.rank == Rank.SKIP:
        state.advance_turn(2)
        return True, "Skip: next player skipped."

    # DRAW2
    if card.rank == Rank.DRAW2:
        if state.manual_mode:
            # Increment target player's count; operator will add their own drawn cards if they are the target.
            if np_index == state.my_index:
                # we can't know the cards; user should add them via UI
                pass
            else:
                state.players[np_index].hidden_count += 2
            state.advance_turn(2)
            return True, f"{state.players[np_index].name} draws 2 (manual)."
        else:
            drawn = _draw_from_deck(state, 2)
            if np_index == state.my_index:
                state.players[np_index].hand.extend(drawn)
            else:
                state.hidden_pool.extend(drawn)
                state.players[np_index].hidden_count += len(drawn)
            state.advance_turn(2)
            return True, f"{state.players[np_index].name} draws 2 and is skipped."

    # WILD+4
    if card.rank == Rank.WILD_DRAW4:
        if state.manual_mode:
            if np_index == state.my_index:
                pass
            else:
                state.players[np_index].hidden_count += 4
            state.advance_turn(2)
            return True, f"{state.players[np_index].name} draws 4 (manual)."
        else:
            drawn = _draw_from_deck(state, 4)
            if np_index == state.my_index:
                state.players[np_index].hand.extend(drawn)
            else:
                state.hidden_pool.extend(drawn)
                state.players[np_index].hidden_count += len(drawn)
            state.advance_turn(2)
            return True, f"{state.players[np_index].name} draws 4 and is skipped."

    # normal / WILD
    state.advance_turn(1)
    return True, "Played."



# adding helper
def manually_add_card_to_my_hand(state: GameState, card: Card) -> Tuple[bool, str]:
    """
    Manual mode helper: operator records the exact card they drew.
    """
    if not state.manual_mode:
        return False, "Only needed in manual mode."
    state.players[state.my_index].hand.append(card)
    return True, f"Added {card.short()} to your hand."
