# streamlit run streamlit_app.py



import streamlit as st
from typing import List, Optional

from uknowuno.cards import Card, Color, Rank
from uknowuno.game_state import GameState
from uknowuno.engine import (
    start_game_with_my_hand,
    legal_moves_for_player,
    play_card_by_index,
    opponent_play_from_menu,
    draw_for_player,
    pass_turn,
)
from uknowuno.strategy import recommend_move

st.set_page_config(page_title="UknowUno", page_icon="🃏", layout="wide")

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

def log(msg: str):
    st.session_state.log.append(msg)
    st.session_state.log = st.session_state.log[-40:]

def reset_selection():
    st.session_state.selected_card_idx = None
    st.session_state.wild_color_pick = Color.RED

init_session()

# ---------------- Lobby ----------------

def lobby_screen():
    st.title("UknowUno 🃏")
    st.subheader("Lobby")
    st.caption("Tell the app your exact 7-card hand, then start. Opponents’ cards stay unknown.")

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.number_input("Number of players", min_value=2, max_value=10, value=4, step=1)
    with c2:
        my_index = st.number_input("Your seat index (0-based)", min_value=0, max_value=int(n)-1, value=0, step=1)
    with c3:
        seed = st.text_input("Shuffle seed (optional)", value="")

    st.write("### Your 7 cards (comma separated)")
    st.caption("Examples: `R-7, B-REVERSE, Y-0, G-2, R-5, WILD, WILD_DRAW4`")
    my_cards_text = st.text_input("Enter exactly 7 cards", value="")

    st.write("### Player names")
    default_names = [f"Player {i}" for i in range(int(n))]
    name_cols = st.columns(5)
    names: List[str] = []
    for i in range(int(n)):
        with name_cols[i % 5]:
            names.append(st.text_input(f"Name {i}", value=default_names[i], key=f"name_{i}"))

    def parse_cards(txt: str) -> Optional[List[Card]]:
        if not txt.strip():
            return None
        parts = [p.strip() for p in txt.split(",") if p.strip()]
        try:
            cards = [Card.from_text(p) for p in parts]
            return cards
        except Exception as e:
            st.error(f"Parse error: {e}")
            return None

    if st.button("Start Game ▶️", type="primary", use_container_width=True):
        cards = parse_cards(my_cards_text) or []
        if len(cards) != 7:
            st.error("Please enter exactly 7 valid cards.")
            return
        seed_int: Optional[int] = None
        if seed.strip():
            try:
                seed_int = int(seed.strip())
            except Exception:
                st.warning("Seed must be an integer; ignoring.")
        try:
            st.session_state.game = start_game_with_my_hand(
                num_players=int(n),
                my_hand=cards,
                my_index=int(my_index),
                names=names,
                seed=seed_int,
                hand_size=7,
            )
        except Exception as e:
            st.error(str(e))
            return
        st.session_state.phase = "table"
        st.session_state.selected_player = int(my_index)
        st.session_state.selected_card_idx = None
        st.session_state.log = ["Game started (your hand recorded)."]
        st.rerun()

# ---------------- Table ----------------

def table_header(game: GameState):
    left, mid, right = st.columns([1,1,1])
    with left:
        st.metric("Deck", f"{len(game.deck)} cards")
        st.metric("Discard", f"{len(game.discard)} cards")
    with mid:
        st.subheader("Top Card")
        if game.top_card:
            st.markdown(f"### {card_icon(game.top_card)}")
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
                label = card_icon(c)
                with col:
                    if st.button(label, key=f"card_{pid}_{idx}", type=("primary" if st.session_state.selected_card_idx == idx else "secondary"), use_container_width=True):
                        st.session_state.selected_card_idx = None if st.session_state.selected_card_idx == idx else idx
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
        # Your seat controls
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

        colA, colB, colC, colD = st.columns([1,1,1,1])
        with colA:
            if st.button("Play", type="primary", use_container_width=True, disabled=(not my_turn or sel_idx is None)):
                chosen = None
                if sel_idx is not None and sel_idx < len(p.hand) and p.hand[sel_idx].is_wild():
                    chosen = st.session_state.wild_color_pick
                ok, msg = play_card_by_index(st.session_state.game, pid, sel_idx if sel_idx is not None else -1, chosen)
                log(f"{p.name} -> Play: {msg}")
                reset_selection()
                st.rerun()

        with colB:
            if st.button("Draw 1", use_container_width=True, disabled=not my_turn):
                ok, msg, drew = draw_for_player(st.session_state.game, pid, 1)
                log(f"{p.name} -> {msg}")
                st.rerun()

        with colC:
            if st.button("Pass", use_container_width=True, disabled=not my_turn):
                ok, msg = pass_turn(st.session_state.game, pid)
                log(f"{p.name} -> {msg}")
                reset_selection()
                st.rerun()

        with colD:
            if st.button("Recommend", use_container_width=True):
                hand = p.hand
                top = game.top_card
                if top is None:
                    st.info("No top card.")
                else:
                    next_size = game.players[game.next_index(1)].total_count()
                    rec = recommend_move(hand, top, game.active_color, next_player_hand_size=next_size)
                    if rec is None:
                        st.info("No recommendation.")
                    else:
                        # highlight the first matching instance
                        idx = None
                        for i, h in enumerate(hand):
                            if h == rec:
                                idx = i
                                break
                        st.session_state.selected_card_idx = idx
                        st.toast(f"Suggested: {card_icon(rec)}")
                        st.rerun()

    else:
        # Opponent seat controls
        legal_only = st.toggle("Show only legal cards (recommended)", value=True)
        options = all_card_options()
        # Filter legal if requested
        if legal_only and game.top_card:
            options = [c for c in options if c.matches(game.top_card, game.active_color)]

        opt_str = [card_icon(c) for c in options]
        choice = st.selectbox("Opponent played...", options=list(range(len(options))), format_func=lambda i: opt_str[i])

        chosen_card = options[choice] if options else None
        if chosen_card and chosen_card.is_wild():
            st.session_state.wild_color_pick = st.radio(
                "Choose color for WILD",
                [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE],
                format_func=lambda x: color_pill(x),
                horizontal=True,
                key="wild_color_opp",
            )

        colA, colB, colC = st.columns([1,1,1])
        with colA:
            if st.button("Record Play", type="primary", use_container_width=True, disabled=(not my_turn or chosen_card is None)):
                chosen = st.session_state.wild_color_pick if (chosen_card and chosen_card.is_wild()) else None
                ok, msg = opponent_play_from_menu(st.session_state.game, pid, chosen_card, chosen)
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

if st.session_state.phase == "lobby":
    lobby_screen()
else:
    table_screen()
