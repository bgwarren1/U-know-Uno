from enum import Enum
from dataclasses import dataclass
from typing import Optional


class Color(str, Enum):
    RED = "R"
    YELLOW = "Y"
    GREEN = "G"
    BLUE = "B"
    WILD = "W"   # for wilds' chosen color


class Rank(str, Enum):
    # numbers
    R0 = "0"
    R1 = "1"
    R2 = "2"
    R3 = "3"
    R4 = "4"
    R5 = "5"
    R6 = "6"
    R7 = "7"
    R8 = "8"
    R9 = "9"
    # actions
    SKIP = "SKIP"
    REVERSE = "REVERSE"
    DRAW2 = "DRAW2"
    # wilds
    WILD = "WILD"
    WILD_DRAW4 = "WILD_DRAW4"


@dataclass(frozen=True) # making the cards immutable
class Card:
    color: Optional[Color]  # None for wild before choosing color
    rank: Rank

    def is_wild(self) -> bool:
        return self.rank in (Rank.WILD, Rank.WILD_DRAW4)

    def matches(self, top: "Card", active_color: Color) -> bool:
        """UNO matching: color match OR rank/action match OR wild."""
        if self.is_wild():
            return True
        if top.is_wild():
            # when top is wild, active_color dictates matches
            return self.color == active_color
        return (self.color == top.color) or (self.rank == top.rank)

    def short(self) -> str:
        if self.is_wild():
            return self.rank.value
        return f"{self.color.value}-{self.rank.value}"

    @staticmethod
    def from_text(s: str) -> "Card":
        """
        Parse human-friendly text into a Card.

        Accepted examples:
        - 'R-7', 'G-REVERSE', 'B-0', 'Y-2'
        - 'WILD', 'WILD_DRAW4'
        - Also accepts long color names: 'RED-7', 'BLUE-REVERSE'
        - Tolerates spaces and case differences.
        """
        s = s.strip().upper()

        # Wilds without color part
        if s in ("WILD", "WILD_DRAW4"):
            return Card(color=None, rank=Rank[s])

        # Must be COLOR-RANK
        if "-" not in s:
            raise ValueError(f"Bad card format: {s}")

        c, r = [part.strip() for part in s.split("-", 1)]

        # Color: allow 'R' or 'RED', etc.
        color = Color[c] if len(c) > 1 else Color(c)

        # Rank:
        #  - If it's a digit '0'..'9', map to Enum name 'R0'..'R9'
        #  - If it's already like 'R7', accept it
        #  - Otherwise it must be an action name (SKIP, REVERSE, DRAW2)
        if r.isdigit():
            rank = Rank[f"R{int(r)}"]
        elif r.startswith("R") and r[1:].isdigit():
            rank = Rank[r]                 # e.g., 'R7'
        else:
            rank = Rank[r]                 # e.g., 'REVERSE', 'DRAW2'

        return Card(color=color, rank=rank)
