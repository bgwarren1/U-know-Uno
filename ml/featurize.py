# from typing import Tuple, Dict, List

# import numpy as np
# from uknowuno.cards import Card, Color, Rank
# from uknowuno.engine import legal_moves_for_player
# from uknowuno.game_state import GameState


# ACTION_VOCAB = []

# # fill in list of actions
# COLOR = [c for c in Color]
# NUM_RANKS = [Rank[f"R{i}"] for i in range(10)]
# ACTIONS = [Rank.SKIP, Rank.REVERSE, Rank.DRAW2]
# WILDS = [Rank.WILD, Rank.WILD_DRAW4]

# # add each action to list
# ACTION_VOCAB = ([Card(c, r) for c in COLOR for r in (NUM_RANKS + ACTIONS)] + [Card(None, r) for r in WILDS])
# A = len(ACTION_VOCAB)


# def encode_state(state: GameState, pid: int) -> Tuple[np.ndarray, np.ndarray, Dict[int, Card]]:
#     # Example features: top-card rank one-hot (13), active color one-hot (4), your hand counts (54),
#     # direction, deck/discard sizes, opponent counts, simple history stats, etc.
#     phi = np.zeros(128, dtype=np.float32)  # chosen size

#     # Build legal mask of size A
#     mask = np.zeros(A, dtype=np.float32)
#     legal = legal_moves_for_player(state, pid)
#     # Turn legal Card objects into action-vocab indices
#     index_to_card: Dict[int, Card] = {}
#     for a_idx, proto in enumerate(ACTION_VOCAB):
#         match = next((c for c in legal if c.rank == proto.rank and (c.is_wild() or c.color == proto.color)), None)
#         if match:
#             mask[a_idx] = 1.0 # marked legal for the mask
#             index_to_card[a_idx] = match

#     return phi, mask, index_to_card

# ml/featurize.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np

from uknowuno.cards import Card, Color, Rank
from uknowuno.engine import legal_moves_for_player
from uknowuno.game_state import GameState

# ----- Action vocabulary (54 actions: 4 colors × (0..9, SKIP, REVERSE, DRAW2) + WILD, WILD_DRAW4)
COLORS = [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]
NUMS = [Rank[f"R{i}"] for i in range(10)]
ACTIONS = [Rank.SKIP, Rank.REVERSE, Rank.DRAW2]
WILDS = [Rank.WILD, Rank.WILD_DRAW4]

ACTION_VOCAB: List[Card] = [Card(c, r) for c in COLORS for r in (NUMS + ACTIONS)] + [Card(None, r) for r in WILDS]

# ----- One-hot helpers
def one_hot_color(c: Optional[Color]) -> np.ndarray:
    v = np.zeros(4, dtype=np.float32)
    if c in COLORS:
        v[COLORS.index(c)] = 1.0
    return v

def one_hot_rank(r: Rank) -> np.ndarray:
    # order: R0..R9, SKIP, REVERSE, DRAW2, WILD, WILD_DRAW4  => 15 dims
    order = NUMS + ACTIONS + WILDS
    v = np.zeros(len(order), dtype=np.float32)
    v[order.index(r)] = 1.0
    return v

def hand_color_counts(hand: List[Card]) -> np.ndarray:
    v = np.zeros(4, dtype=np.float32)
    for c in hand:
        if c.color in COLORS:
            v[COLORS.index(c.color)] += 1.0
    return v

def hand_rank_counts(hand: List[Card]) -> np.ndarray:
    order = NUMS + ACTIONS + WILDS
    v = np.zeros(len(order), dtype=np.float32)
    for c in hand:
        v[order.index(c.rank)] += 1.0
    return v

# at top, after imports
EXPECT_OPPONENTS = 3  # we want to support up to 4 total players => 3 opponents max

def others_counts(state: GameState, me: int) -> np.ndarray:
    """Raw counts for opponents in seating order starting from the next player."""
    out: list[float] = []
    for k in range(1, state.num_players()):
        pid = state.next_index(k)
        if pid == me:
            continue
        p = state.players[pid]
        out.append(len(p.hand) + p.hidden_count)
    return np.array(out, dtype=np.float32)

def others_counts_fixed(state: GameState, me: int, expect: int = EXPECT_OPPONENTS) -> np.ndarray:
    """Pad/truncate opponent counts to a fixed length so feature dim stays constant."""
    raw = others_counts(state, me)
    if raw.size < expect:
        return np.pad(raw, (0, expect - raw.size), mode="constant")
    elif raw.size > expect:
        return raw[:expect]
    return raw

def num_players_one_hot(state: GameState) -> np.ndarray:
    """One-hot for total players in {2,3,4}."""
    v = np.zeros(3, dtype=np.float32)
    n = state.num_players()
    if n in (2, 3, 4):
        v[n - 2] = 1.0
    return v

def pad_mask(state: GameState, expect: int = EXPECT_OPPONENTS) -> np.ndarray:
    """Mask bits indicating which opponent slots are real (1) vs padded (0)."""
    m = np.zeros(expect, dtype=np.float32)
    real = min(expect, state.num_players() - 1)
    m[:real] = 1.0
    return m


def encode_state(state: GameState, me: int) -> np.ndarray:
    top = state.top_card
    assert top is not None, "encode_state requires a top card"
    feats = np.concatenate([
        one_hot_color(state.active_color),          # 4
        one_hot_color(top.color),                   # 4
        one_hot_rank(top.rank),                     # 15
        hand_color_counts(state.players[me].hand),  # 4
        hand_rank_counts(state.players[me].hand),   # 15
        np.array([len(state.players[me].hand)], dtype=np.float32),  # 1
        others_counts_fixed(state, me),             # 3 fixed opponent counts
        pad_mask(state),                            # 3 mask for which slots are real
        num_players_one_hot(state),                 # 3 one-hot (2,3,4 players)
        np.array([
            float(len(state.deck)),                 # 1
            float(len(state.discard)),              # 1
            1.0 if state.direction == 1 else 0.0    # 1
        ], dtype=np.float32),
    ])
    return feats


# ----- Action-only features (type + chosen wild color if any)
def encode_action(card: Card, chosen_color: Optional[Color]) -> np.ndarray:
    return np.concatenate([
        one_hot_color(card.color),        # 4 (wilds => zeros)
        one_hot_rank(card.rank),          # 15
        one_hot_color(chosen_color),      # 4 (all-zero for non-wilds)
        np.array([1.0 if card.is_wild() else 0.0], dtype=np.float32),  # 1
    ])

def build_examples_for_legal_actions(
    state: GameState, me: int
) -> Tuple[List[np.ndarray], List[Tuple[Card, Optional[Color]]]]:
    """Return feature vectors and the corresponding (card, chosen_color) for each legal move.
       For wilds, expand into four color choices."""
    X: List[np.ndarray] = []
    acts: List[Tuple[Card, Optional[Color]]] = []
    base = encode_state(state, me)
    legal = legal_moves_for_player(state, me)
    if not legal:
        return X, acts
    for card in legal:
        if card.is_wild():
            for c in COLORS:
                X.append(np.concatenate([base, encode_action(card, c)]))
                acts.append((card, c))
        else:
            X.append(np.concatenate([base, encode_action(card, None)]))
            acts.append((card, None))
    return X, acts

def feature_dim(state: GameState) -> int:
    # convenience to see final dimensionality
    x, _ = build_examples_for_legal_actions(state, state.current_player)
    return len(x[0]) if x else 0

