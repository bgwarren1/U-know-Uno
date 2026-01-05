# streamlit run streamlit_app.py

import random

import streamlit as st
from typing import List, Optional

from ml.infer_xgb import load_xgb_for_players, pick_with_xgb
 
import copy
from ml.rollout_oracle import estimate_baseline, map_pid_by_name

from uknowuno.cards import Card, Color, Rank
from uknowuno.game_state import GameState
from uknowuno.engine import (
    start_game_with_my_hand,
    legal_moves_for_player,
    play_card_by_index,
    opponent_play_from_menu,
    draw_for_player,
    pass_turn,
    manually_add_card_to_my_hand,   
)

from ml.rollout_oracle import evaluate_current_position, find_hand_index_of_card, determinize_from_counts, evaluate_ensemble

from uknowuno.rules import full_deck

from uknowuno.strategy import recommend_move




# ---------------- XGBOOST (cache per player-count) ----------------
@st.cache_resource
def _load_model_for(n_players: int):
    try:
        # Loads models/xgb_{n}p.json + sidecar meta, saved as Booster
        return load_xgb_for_players(n_players)
    except Exception:
        return None


st.markdown("""
<style>
/* make the little pick buttons not chunky */
.stButton > button {
  padding: 0.15rem 0.25rem !important;
  line-height: 1 !important;
  border-radius: 10px !important;
}

/* reduce vertical gaps a bit */
div[data-testid="stVerticalBlock"] > div { gap: 0.35rem; }
</style>
""", unsafe_allow_html=True)


st.set_page_config(page_title="UknowUno", page_icon="🃏", layout="wide")






# Actual card image UI helpers


import os

# ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "cards")
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


COLOR_TO_PREFIX = {
    Color.RED: "red",
    Color.BLUE: "blue",
    Color.GREEN: "green",
    Color.YELLOW: "yellow",
}

NUM_RANKS = {Rank.R0, Rank.R1, Rank.R2, Rank.R3, Rank.R4, Rank.R5, Rank.R6, Rank.R7, Rank.R8, Rank.R9}

def card_png_path(c: Card) -> str:
    # wilds
    if c.rank == Rank.WILD:
        return os.path.join(ASSET_DIR, "wild.png")
    if c.rank == Rank.WILD_DRAW4:
        return os.path.join(ASSET_DIR, "wild_draw4.png")

    prefix = COLOR_TO_PREFIX[c.color]

    # numbers (ONLY these, not REVERSE)
    if c.rank in NUM_RANKS:
        n = int(c.rank.name[1:])  # "R7" -> 7
        return os.path.join(ASSET_DIR, f"{prefix}_{n}.png")

    # actions
    if c.rank == Rank.SKIP:
        return os.path.join(ASSET_DIR, f"{prefix}_skip.png")
    if c.rank == Rank.REVERSE:
        return os.path.join(ASSET_DIR, f"{prefix}_reverse.png")
    if c.rank == Rank.DRAW2:
        return os.path.join(ASSET_DIR, f"{prefix}_draw2.png")

    # fallback
    return os.path.join(ASSET_DIR, "wild.png")


def cards_for_section(section: str) -> List[Card]:
    section = section.lower()
    if section == "wild":
        return [Card(None, Rank.WILD), Card(None, Rank.WILD_DRAW4)]

    color_map = {
        "red": Color.RED,
        "blue": Color.BLUE,
        "green": Color.GREEN,
        "yellow": Color.YELLOW,
    }
    col = color_map[section]
    nums = [Rank[f"R{i}"] for i in range(10)]
    acts = [Rank.SKIP, Rank.REVERSE, Rank.DRAW2]
    return [Card(col, r) for r in (nums + acts)]


def image_section_picker(key_prefix: str, label: str = "Pick a card", cols: int = 7, img_width: int = 70) -> Card:
    st.caption(label)

    # section state
    sec_key = f"{key_prefix}_section"
    if sec_key not in st.session_state:
        st.session_state[sec_key] = "Red"

    # section buttons
    sections = ["Red", "Blue", "Green", "Yellow", "Wild"]
    sec_cols = st.columns(5)
    for i, s in enumerate(sections):
        with sec_cols[i]:
            if st.button(
                s,
                key=f"{key_prefix}_secbtn_{s}",
                use_container_width=True,
                type="primary" if st.session_state[sec_key] == s else "secondary",
            ):
                st.session_state[sec_key] = s
                st.rerun()

    # grid
    candidates = cards_for_section(st.session_state[sec_key])
    grid = st.columns(cols)
    pick_key = f"{key_prefix}_picked"

    # default picked
    if pick_key not in st.session_state:
        st.session_state[pick_key] = candidates[0].short()

    for i, card in enumerate(candidates):
        with grid[i % cols]:
            st.image(card_png_path(card), width=img_width)

            is_sel = (st.session_state[pick_key] == card.short())
            if st.button(
                "✓" if is_sel else "",
                key=f"{key_prefix}_pickbtn_{card.short()}",
                use_container_width=True,
                type="primary" if is_sel else "secondary",
            ):
                st.session_state[pick_key] = card.short()
                st.rerun()

    return Card.from_text(st.session_state[pick_key])



# ---------------- UI helpers ----------------

COLOR_EMOJI = {
    Color.RED: "🟥",
    Color.YELLOW: "🟨",
    Color.GREEN: "🟩",
    Color.BLUE: "🟦",
}
def card_icon(c: Card) -> str:
    if c.is_wild():
        return "🃏 " + ("+4" if c.rank == Rank.WILD_DRAW4 else "WILD")
    if c.rank.name.startswith("R"):
        return f"{COLOR_EMOJI.get(c.color, '⬜')} {c.rank.value[-1]}"
    return f"{COLOR_EMOJI.get(c.color, '⬜')} {c.rank.value}"

def color_pill(c: Color) -> str:
    return f"{COLOR_EMOJI[c]} {c.name.title()}"

def all_card_options() -> List[Card]:
    options: List[Card] = []
    colors = [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE]
    numbers = [Rank.R0, Rank.R1, Rank.R2, Rank.R3, Rank.R4, Rank.R5, Rank.R6, Rank.R7, Rank.R8, Rank.R9]
    actions = [Rank.SKIP, Rank.REVERSE, Rank.DRAW2]
    for col in colors:
        for r in numbers + actions:
            options.append(Card(col, r))
    options.extend([Card(None, Rank.WILD), Card(None, Rank.WILD_DRAW4)])
    return options

def init_session():
    if "phase" not in st.session_state:
        st.session_state.phase = "lobby"
    if "game" not in st.session_state:
        st.session_state.game: Optional[GameState] = None
    if "selected_player" not in st.session_state:
        st.session_state.selected_player = 0
    if "selected_card_idx" not in st.session_state:
        st.session_state.selected_card_idx = None
    if "wild_color_pick" not in st.session_state:
        st.session_state.wild_color_pick = Color.RED
    if "log" not in st.session_state:
        st.session_state.log: List[str] = []
    if "winner_pid" not in st.session_state:
        st.session_state.winner_pid = None
    if "ml_pending_card_short" not in st.session_state:
        st.session_state.ml_pending_card_short = None  # str like "WILD" or "R-5"
    if "ml_pending_color" not in st.session_state:
        st.session_state.ml_pending_color = None  # Color or None



# Detect a winner of the game; 0 card count
def detect_winner_pid(game: GameState) -> Optional[int]:
    # winner = first player with 0 total cards
    for pid, p in enumerate(game.players):
        if p.total_count() == 0:
            return pid
    return None



def log(msg: str):
    st.session_state.log.append(msg)
    st.session_state.log = st.session_state.log[-40:]

def reset_selection():
    st.session_state.selected_card_idx = None
    st.session_state.wild_color_pick = Color.RED



def compute_deck_remaining(game: GameState) -> tuple[int, int, int, int]:
    """Return (remaining_est, total_cards, in_hands, in_discard).
    remaining_est works for both manual and non-manual modes."""
    total_cards = len(full_deck())  # robust to custom decks
    in_hands = sum(len(p.hand) + getattr(p, "hidden_count", 0) for p in game.players)
    in_discard = len(game.discard)
    remaining_est = max(total_cards - in_hands - in_discard, 0)
    return remaining_est, total_cards, in_hands, in_discard

def deck_indicator(game: GameState):
    remaining_est, total_cards, in_hands, in_discard = compute_deck_remaining(game)

    if not game.manual_mode:
        # Engine also tracks an actual deck size; show both and reconcile if needed.
        engine_left = len(game.deck)
        if engine_left != remaining_est:
            st.metric("Deck", f"{engine_left} (est {remaining_est})")
            st.caption(
                f"Accounting: total {total_cards} − hands {in_hands} − discard {in_discard} = {remaining_est}"
            )
        else:
            st.metric("Deck", f"{engine_left}")
            st.caption(f"Discard: {in_discard} • In hands: {in_hands}")
    else:
        # Manual mode: no engine-owned draw pile, but we can still compute it.
        st.metric("Deck", f"{remaining_est}")
        st.caption(
            f"Computed from accounting: total {total_cards} − hands {in_hands} − discard {in_discard}"
        )



def nested_card_picker(key_prefix: str, label: str = "Pick a card") -> Card:
    st.caption(label)
    group = st.radio(
        "Color group",
        ["Red", "Yellow", "Green", "Blue", "Wild"],
        horizontal=True,
        key=f"{key_prefix}_group",
    )
    # map to Color for non-wild
    color_map = {
        "Red": Color.RED,
        "Yellow": Color.YELLOW,
        "Green": Color.GREEN,
        "Blue": Color.BLUE,
    }
    if group == "Wild":
        wild_type = st.radio(
            "Wild type",
            ["WILD", "WILD_DRAW4"],
            horizontal=True,
            key=f"{key_prefix}_wildtype",
        )
        rank = Rank.WILD if wild_type == "WILD" else Rank.WILD_DRAW4
        return Card(None, rank)
    else:
        val = st.selectbox(
            "Value / Action",
            ["0","1","2","3","4","5","6","7","8","9","SKIP","REVERSE","DRAW2"],
            key=f"{key_prefix}_value",
        )
        if val.isdigit():
            rank = Rank[f"R{val}"]
        else:
            rank = Rank[val]  # SKIP/REVERSE/DRAW2
        return Card(color_map[group], rank)




init_session()







# ---------------- Lobby ----------------

def lobby_screen():
    st.title("UknowUno")
    st.subheader("Lobby")
    st.caption("Tell the app your exact 7-card hand, then pick a starting top card. Opponents stay hidden.")
    manual_mode = st.toggle(
        "Manual Mode (you will input every played card; no deck)",
        value=st.session_state.get("manual_mode", True),
        key="lobby_manual_mode_toggle",   # <<< unique key fixes the collision
    )
    st.session_state.manual_mode = manual_mode  # keep it sticky on reruns



    # --- basic setup inputs ---
    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.number_input("Number of players", min_value=2, max_value=10, value=4, step=1)
    with c2:
        my_index = st.number_input("Your seat index (0 for 1st, 1 for second, etc.)", min_value=0, max_value=int(n)-1, value=0, step=1)
    with c3:
        seed = st.text_input("Shuffle seed (optional)", value="")




    # --- your hand input ---
    # st.write("### Your 7 cards (comma separated)")
    # st.caption("Examples: `R-7, B-REVERSE, Y-0, G-2, R-5, WILD, WILD_DRAW4`")
    # my_cards_text = st.text_input("Enter exactly 7 cards", value="", key="lobby_my_hand")
    st.write("### Build your 7-card hand")
    st.caption("Click cards to add them. Use Clear/Undo if you misclick.")

    # session storage for lobby picks (UI only)
    if "lobby_hand" not in st.session_state:
        st.session_state.lobby_hand = []  # list[str] of Card.short()
    if "lobby_section" not in st.session_state:
        st.session_state.lobby_section = "Red"

    sec_cols = st.columns(5)
    sections = ["Red", "Blue", "Green", "Yellow", "Wild"]
    for i, s in enumerate(sections):
        with sec_cols[i]:
            if st.button(s, key=f"lobby_sec_{s}", use_container_width=True,
                        type="primary" if st.session_state.lobby_section == s else "secondary"):
                st.session_state.lobby_section = s
                st.rerun()

    # show current hand as images + count
    st.write(f"**Current hand:** {len(st.session_state.lobby_hand)}/7")
    hand_cols = st.columns(7)
    for i in range(7):
        with hand_cols[i]:
            if i < len(st.session_state.lobby_hand):
                c = Card.from_text(st.session_state.lobby_hand[i])
                st.image(card_png_path(c), use_container_width=True)
            else:
                st.caption("—")

    ctrlA, ctrlB, ctrlC = st.columns([1,1,2])
    with ctrlA:
        if st.button("Undo", use_container_width=True, disabled=(len(st.session_state.lobby_hand) == 0)):
            st.session_state.lobby_hand.pop()
            st.rerun()
    with ctrlB:
        if st.button("Clear", use_container_width=True, disabled=(len(st.session_state.lobby_hand) == 0)):
            st.session_state.lobby_hand = []
            st.rerun()

    st.write("---")
    st.write(f"**Pick from {st.session_state.lobby_section}:**")

    # grid of selectable cards (image + button)
    candidates = cards_for_section(st.session_state.lobby_section)
    grid = st.columns(7)
    for i, card in enumerate(candidates):
        with grid[i % 7]:
            st.image(card_png_path(card), use_container_width=True)
            disabled = (len(st.session_state.lobby_hand) >= 7)
            if st.button("Add", key=f"lobby_add_{st.session_state.lobby_section}_{card.short()}",
                        use_container_width=True, disabled=disabled):
                st.session_state.lobby_hand.append(card.short())
                st.rerun()

    # This is the string your existing parse_my_hand can use (no format change)
    my_cards_text = ", ".join(st.session_state.lobby_hand)



    """
    Adding Actual Uno card icons for starting top card
    """
    if "lobby_top" not in st.session_state:
        st.session_state.lobby_top = ""  # stores Card.short() like "R-5" or "WILD"
    if "lobby_top_section" not in st.session_state:
        st.session_state.lobby_top_section = "Red"

    # --- optional starting top card inputs ---
    # st.write("### (Optional) Starting top card")
    # st.caption("Examples: `R-5`, `G-REVERSE`, `WILD`, `WILD_DRAW4`")
    # start_top_text = st.text_input("Starting top card (optional)", value="", key="lobby_start_top_text")
    st.write("### Starting top card")
    top_on = st.toggle(
        "Click for card menu",
        value=(st.session_state.lobby_top != ""),
        key="lobby_top_toggle",
    )

    if not top_on:
        st.session_state.lobby_top = ""
        start_top_text = ""   # keep your existing parsing path happy
    else:
        # section buttons
        sec_cols = st.columns(5)
        sections = ["Red", "Blue", "Green", "Yellow", "Wild"]
        for i, s in enumerate(sections):
            with sec_cols[i]:
                if st.button(
                    s,
                    key=f"top_sec_{s}",
                    use_container_width=True,
                    type="primary" if st.session_state.lobby_top_section == s else "secondary",
                ):
                    st.session_state.lobby_top_section = s
                    st.rerun()

        # grid of candidates for that section
        st.write(f"**Pick from {st.session_state.lobby_top_section}:**")
        candidates = cards_for_section(st.session_state.lobby_top_section)
        grid = st.columns(7)
        for i, card in enumerate(candidates):
            with grid[i % 7]:
                st.image(card_png_path(card), use_container_width=True)
                is_sel = (st.session_state.lobby_top == card.short())
                if st.button(
                    "✓" if is_sel else "Set",
                    key=f"top_set_{st.session_state.lobby_top_section}_{card.short()}",
                    use_container_width=True,
                    type="primary" if is_sel else "secondary",
                ):
                    st.session_state.lobby_top = card.short()
                    st.rerun()

    # preview + bridge to your existing parse path
    start_top_text = st.session_state.lobby_top
    if start_top_text:
        st.caption("Selected top card:")
        st.image(card_png_path(Card.from_text(start_top_text)), width=140)

    # optional: clear button
    if st.button("Clear top card", key="top_clear", use_container_width=True):
        st.session_state.lobby_top = ""
        st.rerun()


    # Try to parse the starting top immediately so we can show a color picker if it's a wild
    parsed_start_top = None
    parse_err = None
    txt = start_top_text.strip()
    if txt:
        try:
            parsed_start_top = Card.from_text(txt)
        except Exception as e:
            parse_err = str(e)

    # If wild, let the user pick the active color for the wild top
    initial_active_color = None
    if parsed_start_top is not None and parsed_start_top.is_wild():
        initial_active_color = st.radio(
            "Active color for starting WILD",
            [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE],
            format_func=lambda c: c.name.title(),
            horizontal=True,
            key="lobby_wild_color_pick",
        )

    


    if parsed_start_top is not None:
        st.caption(f"Parsed starting top: {card_icon(parsed_start_top)}")
    elif txt and parse_err:
        st.warning(f"Could not parse starting top card yet: {parse_err}")

    # --- names input ---
    st.write("### Player names")
    default_names = [f"Player {i}" for i in range(int(n))]
    name_cols = st.columns(5)
    names: List[str] = []
    for i in range(int(n)):
        with name_cols[i % 5]:
            names.append(st.text_input(f"Name {i}", value=default_names[i], key=f"name_{i}"))

    # HELPERS
    def parse_my_hand(txt: str) -> Optional[List[Card]]:
        if not txt.strip():
            return None
        parts = [p.strip() for p in txt.split(",") if p.strip()]
        try:
            cards = [Card.from_text(p) for p in parts]
            return cards
        except Exception as e:
            st.error(f"Hand parse error: {e}")
            return None



    # START GAME
    if st.button("Start Game ▶️", type="primary", use_container_width=True):
        # 1) Your 7 cards
        my_cards = parse_my_hand(my_cards_text) or []
        if len(my_cards) != 7:
            st.error("Please enter exactly 7 valid cards for your hand.")
            return

        # 2) Optional starting top (validate only if user provided text)
        initial_top = None
        if txt:
            if parsed_start_top is None:
                st.error(f"Could not parse starting top card: {parse_err or 'invalid format'}")
                return
            initial_top = parsed_start_top

        # 3) Optional seed
        seed_int: Optional[int] = None
        if seed.strip():
            try:
                seed_int = int(seed.strip())
            except Exception:
                st.warning("Seed must be an integer; ignoring.")

        # 4) Create game
        try:
            st.session_state.game = start_game_with_my_hand(
                num_players=int(n),
                my_hand=my_cards,
                my_index=int(my_index),
                names=names,
                seed=seed_int,
                hand_size=7,                      
                initial_top=initial_top,                       # <— NEW
                initial_active_color=initial_active_color,     # <— NEW (only used if wild)
                manual_mode=manual_mode,
            )
        except Exception as e:
            st.error(str(e))
            return

        # 5) Move to table
        st.session_state.phase = "table"
        st.session_state.selected_player = int(my_index)
        st.session_state.selected_card_idx = None
        st.session_state.log = ["Game started" + (f" with top {initial_top.short()}" if initial_top else ".")]
        st.rerun()


# ---------------- Table ----------------

def table_header(game: GameState):
    left, mid, right = st.columns([1,1,1])
    with left:
        deck_indicator(game)
        st.metric("Discard", f"{len(game.discard)} cards")
    with mid:
        
        st.subheader("Top Card")
        if game.top_card:
            st.image(card_png_path(game.top_card), width=110)
        st.caption(f"Active color: **{color_pill(game.active_color)}**")
    with right:
        direction = "Clockwise ↻" if game.direction == 1 else "Counterclockwise ↺"
        st.metric("Turn", f"{game.players[game.current_player].name}")
        st.metric("Direction", direction)

def render_player_seat(game: GameState, pid: int):
    p = game.players[pid]
    my_seat = (pid == game.my_index)
    my_turn = (pid == game.current_player)
    selected = (pid == st.session_state.selected_player)

    box = st.container(border=True)
    with box:
        header = f"🎴 {p.name}  ({p.total_count()} cards)"
        if my_turn:
            header += "  – **Your turn**" if my_seat else "  – **Their turn**"
        if selected:
            header += "  – **Selected**"
        st.markdown(header)

        # Controls to select this player
        sel_cols = st.columns([1,1,2])
        with sel_cols[0]:
            if st.button("Select", key=f"select_{pid}", use_container_width=True):
                st.session_state.selected_player = pid
                reset_selection()
                st.rerun()
        with sel_cols[1]:
            if st.button("UNO!", key=f"uno_{pid}", use_container_width=True):
                p.said_uno = not p.said_uno
                log(f"{p.name} toggled UNO to {p.said_uno}.")
                st.rerun()

        st.write("---")

        # Show hands: YOU see your cards; opponents show backs only.
        if my_seat:
            # clickable cards
            card_cols = st.columns(8, vertical_alignment="center")
            for idx, c in enumerate(p.hand):
                col = card_cols[idx % 8]
                with col:
                    st.image(card_png_path(c), width=55)  # small icon size
                    is_sel = (st.session_state.selected_card_idx == idx)
                    if st.button("✓" if is_sel else "", key=f"card_{pid}_{idx}",
                                type="primary" if is_sel else "secondary",
                                use_container_width=True):
                        st.session_state.selected_card_idx = None if is_sel else idx
                        st.rerun()
        else:
            st.write("🂠 " * min(p.total_count(), 20))
            if p.total_count() > 20:
                st.caption(f"+{p.total_count()-20} more")

        # Only show "legal moves" hint for your seat
        if my_seat and game.top_card:
            lm = legal_moves_for_player(game, pid)
            if lm:
                st.caption("Legal now: " + ", ".join([card_icon(c) for c in lm]))
            else:
                st.caption("_No legal moves (draw or pass)_")

def action_panel(game: GameState):
    pid = st.session_state.selected_player
    p = game.players[pid]
    my_seat = (pid == game.my_index)
    my_turn = (pid == game.current_player)

    st.subheader("Actions")

    if my_seat:
        # ---- Your seat controls -------------------------------------------------
        sel_idx = st.session_state.selected_card_idx
        if sel_idx is not None and sel_idx < len(p.hand):
            c = p.hand[sel_idx]
            st.write(f"Selected: **{card_icon(c)}**")
            if c.is_wild():
                st.session_state.wild_color_pick = st.radio(
                    "Choose color for WILD",
                    [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE],
                    format_func=lambda x: color_pill(x),
                    horizontal=True,
                    key="wild_color_me",
                )
        else:
            st.caption("Click one of your cards to select it.")

        colA, colB, colC, colD = st.columns([1, 1, 1, 1])

        with colA:
            if st.button("Play", type="primary", use_container_width=True, disabled=(not my_turn or sel_idx is None)):
                chosen = None
                if sel_idx is not None and sel_idx < len(p.hand) and p.hand[sel_idx].is_wild():
                    # If ML recommended THIS wild and picked a color, use it automatically
                    if (
                        st.session_state.get("ml_pending_card_short") == p.hand[sel_idx].short()
                        and st.session_state.get("ml_pending_color") is not None
                    ):
                        chosen = st.session_state.get("ml_pending_color")
                    else:
                        chosen = st.session_state.get("wild_color_pick", None)

                ok, msg = play_card_by_index(
                    st.session_state.game,
                    pid,
                    sel_idx if sel_idx is not None else -1,
                    chosen
                )
                log(f"{p.name} -> Play: {msg}")
                reset_selection()

                # Clear ML pending info so it doesn't "stick" to future plays
                st.session_state.ml_pending_card_short = None
                st.session_state.ml_pending_color = None

                st.rerun()


        with colB:
            draw_disabled = (not my_turn) or st.session_state.game.manual_mode
            if st.button("Draw 1", use_container_width=True, disabled=draw_disabled):
                ok, msg, _ = draw_for_player(st.session_state.game, pid, 1)
                log(f"{p.name} -> {msg}")
                st.rerun()

        with colC:
            if st.button("Pass", use_container_width=True, disabled=not my_turn):
                ok, msg = pass_turn(st.session_state.game, pid)
                log(f"{p.name} -> {msg}")
                reset_selection()
                st.rerun()

        with colD:
            if st.button("Recommend (ML)", use_container_width=True):
                top = game.top_card
                if top is None:
                    st.info("No top card.")
                else:
                    n_players = st.session_state.game.num_players()
                    model = _load_model_for(n_players)

                    if model is None:
                        st.warning(f"No ML model found for {n_players} players.")
                    else:
                        card, color, scored = pick_with_xgb(model, st.session_state.game, pid)
                        if card is None:
                            st.warning("No legal actions.")
                        else:
                            idx = find_hand_index_of_card(p.hand, card)
                            if idx is None:
                                st.warning("ML recommended a card not found in your hand (state changed). Try again.")
                            else:
                                st.session_state.selected_card_idx = idx
                                st.session_state.ml_pending_color = color  # may be None
                                st.session_state.ml_pending_card_short = card.short()

                                label = card.short() + (f" → {color.name.title()}" if color else "")
                                st.toast(f"ML suggests: {label}")
                                st.rerun()


        

        # ---- ⚡ Instant ML recommender (XGBoost) --------------------------------
        with st.expander("⚡ Instant ML recommender (XGBoost)"):
            n_players = st.session_state.game.num_players()
            model = _load_model_for(n_players)
            if model is None:
                st.info(
                    f"No model found for {n_players} players. "
                    "Train it with:  "
                    f"`python -m ml.train_xgb --players {n_players} --games 20000 --rollouts 64 --seed 42 --out models/xgb_{n_players}p.json` "
                    "then rerun the app."
                )
            else:
                if st.button("Recommend with ML", use_container_width=True, key=f"ml_rec_{pid}"):
                    card, color, scored = pick_with_xgb(model, st.session_state.game, pid)
                    if card is None:
                        st.warning("No legal actions.")
                    else:
                        label = card.short() + (f" → {color.name.title()}" if color else "")
                        st.success(f"ML suggests: {label}")
                        idx = find_hand_index_of_card(st.session_state.game.players[pid].hand, card)
                        if st.button("Play ML suggestion", key=f"ml_play_{pid}"):
                            ok, msg = play_card_by_index(st.session_state.game, pid, idx or 0, color)
                            log(f"{p.name} (ML) -> {msg}")
                            st.rerun()


        # ---- 🤖 Rollout Oracle (ensemble) --------------------------------------
        with st.expander("Want to look deeper? Estimate win rate for each move"):
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                rollouts = st.slider("Rollouts per action", 8, 256, 64, step=8, key=f"oracle_roll_{pid}")
            with c2:
                worlds = st.slider("Worlds", 2, 32, 8, step=1, key=f"oracle_worlds_{pid}")
            with c3:
                base_seed = st.number_input("Base seed", value=123, step=1, key=f"oracle_seed_{pid}")

            resample_toggle = st.checkbox(
                "Resample hidden info (non-manual)",
                value=not st.session_state.game.manual_mode,
                key=f"oracle_resample_{pid}",
            )

            if st.button("Run Oracle", type="primary", use_container_width=True, key=f"run_oracle_{pid}"):
                ests = evaluate_ensemble(
                    st.session_state.game,
                    my_id=pid,
                    n_worlds=int(worlds),
                    n_rollouts_per_action=int(rollouts),
                    rng_seed=int(base_seed),
                    force_determinize=resample_toggle or st.session_state.game.manual_mode,
                )
                # --- Optional baseline display (sanity check) ---
                try:
                    # Map my seat in this current state (in case you ever reorder players)
                    my_id_world = map_pid_by_name(st.session_state.game, st.session_state.game, pid)
                    base = estimate_baseline(copy.deepcopy(st.session_state.game), my_id_world, trials=128)
                    st.caption(f"Baseline win (no forced action): ~{base:.3f}")
                except Exception as e:
                    # Keep the UI resilient if anything goes wrong
                    st.caption(f"Baseline unavailable: {e}")

                st.session_state[f"oracle_results_{pid}"] = ests

            ests = st.session_state.get(f"oracle_results_{pid}", [])
            if ests:
                st.write("**Top moves (aggregated across worlds):**")
                for i, e in enumerate(ests[:5], 1):
                    label = e.card.short()
                    if e.card.is_wild() and e.chosen_color:
                        label += f" → {e.chosen_color.name.title()}"
                    n = max(e.trials, 1)
                    p_hat = e.win_rate
                    se = (p_hat * (1 - p_hat) / n) ** 0.5
                    lo, hi = max(0.0, p_hat - 1.96 * se), min(1.0, p_hat + 1.96 * se)
                    st.write(f"{i}. {label} — win≈ **{p_hat:.3f}**  (±{1.96*se:.3f})  [{lo:.3f}, {hi:.3f}]  (wins {e.wins}/{e.trials})")


        # ---- Manual add card (only in manual mode) ------------------------------
        if st.session_state.game.manual_mode:
            st.write("---")
            st.markdown("**Manual draw / add card to your hand**")
            my_new_card = image_section_picker("me_add", label="Card you drew", cols=7, img_width=65)

            if my_new_card.is_wild():
                st.caption("Wilds in hand are colorless until you play them (you'll pick the color on play).")
            if st.button("Add to my hand", use_container_width=True):
                ok, msg = manually_add_card_to_my_hand(st.session_state.game, my_new_card)
                log(f"You -> {msg}")
                st.rerun()

    else:
        # ---- Opponent seat controls --------------------------------------------
        played_card = image_section_picker(f"opp_{pid}", label="Opponent played...", cols=7, img_width=65)

        chosen_color = None
        if played_card.is_wild():
            chosen_color = st.radio(
                "Choose color for WILD",
                [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE],
                format_func=lambda x: f"{x.name.title()}",
                horizontal=True,
                key=f"opp_wild_color_{pid}",
            )

        colA, colB, colC = st.columns([1, 1, 1])

        with colA:
            if st.button("Record Play", type="primary", use_container_width=True, disabled=not my_turn):
                ok, msg = opponent_play_from_menu(st.session_state.game, pid, played_card, chosen_color)
                log(f"{p.name} -> Play: {msg}")
                st.rerun()

        with colB:
            if st.button("Draw 1", use_container_width=True, disabled=not my_turn):
                ok, msg, _ = draw_for_player(st.session_state.game, pid, 1)
                log(f"{p.name} -> {msg}")
                st.rerun()

        with colC:
            if st.button("Pass", use_container_width=True, disabled=not my_turn):
                ok, msg = pass_turn(st.session_state.game, pid)
                log(f"{p.name} -> {msg}")
                st.rerun()


def history_panel():
    st.subheader("History")
    if not st.session_state.log:
        st.caption("_Empty_")
    else:
        for line in reversed(st.session_state.log):
            st.write("• " + line)

def table_screen():
    st.title("UknowUno 🃏")
    game: Optional[GameState] = st.session_state.game
    if game is None:
        st.session_state.phase = "lobby"
        st.rerun()
    
    winner = detect_winner_pid(game)
    if winner is not None:
        st.session_state.winner_pid = winner
        st.session_state.phase = "game_over"
        st.rerun()


    table_header(game)  # type: ignore[arg-type]
    st.write("---")

    # Seats: two rows
    n = game.num_players()
    top_row = (n // 2)
    bottom_row = n - top_row

    if top_row > 0:
        cols = st.columns(top_row)
        for i in range(top_row):
            with cols[i]:
                render_player_seat(game, i)

    if bottom_row > 0:
        cols = st.columns(bottom_row)
        for j in range(bottom_row):
            with cols[j]:
                render_player_seat(game, top_row + j)

    st.write("---")
    left, right = st.columns([3,1])
    with left:
        action_panel(game)
    with right:
        history_panel()

    st.write("---")
    if st.button("⏹ End game & back to lobby", use_container_width=True):
        st.session_state.phase = "lobby"
        st.session_state.game = None
        st.session_state.log = []
        st.rerun()

# ---------------- Router ----------------


# Game over screen function
def game_over_screen():
    st.title("UknowUno 🃏")
    st.header("🎉 Game Over")

    game: Optional[GameState] = st.session_state.game
    winner_pid = st.session_state.get("winner_pid", None)

    if game is None or winner_pid is None:
        st.info("No game result found. Returning to lobby.")
        st.session_state.phase = "lobby"
        st.rerun()
        return

    winner_name = game.players[winner_pid].name
    st.markdown(f"## **{winner_name} wins!** 🏆")

    st.write("---")
    c1, c2 = st.columns([1, 1])

    with c1:
        if st.button("⬅️ Back to Lobby", type="primary", use_container_width=True):
            st.session_state.phase = "lobby"
            st.session_state.game = None
            st.session_state.log = []
            st.session_state.winner_pid = None
            # optional: reset lobby pickers if you want:
            # st.session_state.lobby_hand = []
            # st.session_state.lobby_top = ""
            st.rerun()

    with c2:
        if st.button("🔁 Keep viewing final table", use_container_width=True):
            st.session_state.phase = "table"
            st.rerun()

    # optional: show last moves
    with st.expander("Show game history"):
        if not st.session_state.log:
            st.caption("_Empty_")
        else:
            for line in reversed(st.session_state.log):
                st.write("• " + line)




if st.session_state.phase == "lobby":
    lobby_screen()
elif st.session_state.phase == "table":
    table_screen()
else:  # "game_over"
    game_over_screen()




