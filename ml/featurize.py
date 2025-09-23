from typing import Tuple, Dict, List

import numpy as np
from uknowuno.cards import Card, Color, Rank
from uknowuno.engine import legal_moves_for_player
from uknowuno.game_state import GameState


ACTION_VOCAB = []

# fill in list of actions
COLOR = [c for c in Color]
NUM_RANKS = [Rank[f"R{i}"] for i in range(10)]
ACTIONS = [Rank.SKIP, Rank.REVERSE, Rank.DRAW2]
WILDS = [Rank.WILD, Rank.WILD_DRAW4]

# add each action to list
ACTION_VOCAB = ([Card(c, r) for c in COLOR for r in (NUM_RANKS + ACTIONS)] + [Card(None, r) for r in WILDS])
A = len(ACTION_VOCAB)


def encode_state(state: GameState, pid: int) -> Tuple[np.ndarray, np.ndarray, Dict[int, Card]]:
    # Example features: top-card rank one-hot (13), active color one-hot (4), your hand counts (54),
    # direction, deck/discard sizes, opponent counts, simple history stats, etc.
    phi = np.zeros(128, dtype=np.float32)  # chosen size

    # Build legal mask of size A
    mask = np.zeros(A, dtype=np.float32)
    legal = legal_moves_for_player(state, pid)
    # Turn legal Card objects into action-vocab indices
    index_to_card: Dict[int, Card] = {}
    for a_idx, proto in enumerate(ACTION_VOCAB):
        match = next((c for c in legal if c.rank == proto.rank and (c.is_wild() or c.color == proto.color)), None)
        if match:
            mask[a_idx] = 1.0 # marked legal for the mask
            index_to_card[a_idx] = match

    return phi, mask, index_to_card


