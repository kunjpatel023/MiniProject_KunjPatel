import streamlit as st
import random

st.set_page_config(page_title="Tic-Tac-Toe", page_icon="❌", layout="centered")

st.title("❌ Tic-Tac-Toe ⭕")
st.markdown("""
Welcome to Tic-Tac-Toe! Choose your game mode:  
- **1 Player**: Play against a simple computer opponent.  
- **2 Players**: Take turns with a friend. Enjoy!
""")

if "board" not in st.session_state:
    st.session_state.board = [""] * 9  
    st.session_state.current_player = "X"
    st.session_state.winner = None
    st.session_state.game_mode = None

st.subheader("Game Mode")
game_mode = st.selectbox("Choose mode", ["1 Player (vs Computer)", "2 Players"], key="mode_select")


if st.session_state.game_mode != game_mode:
    st.session_state.game_mode = game_mode
    st.session_state.board = [""] * 9
    st.session_state.current_player = "X"
    st.session_state.winner = None


def check_winner(board):
    winning_combinations = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  
        (0, 4, 8), (2, 4, 6)              
    ]
    for a, b, c in winning_combinations:
        if board[a] == board[b] == board[c] and board[a] != "":
            return board[a]
    if "" not in board:
        return "Draw"
    return None

def computer_move(board):
    available = [i for i, spot in enumerate(board) if spot == ""]
    if available:
        return random.choice(available)
    return None


def make_move(index):
    if st.session_state.board[index] == "" and not st.session_state.winner:
        st.session_state.board[index] = st.session_state.current_player
        st.session_state.winner = check_winner(st.session_state.board)
        
        if not st.session_state.winner:
            if st.session_state.game_mode == "2 Players":
                st.session_state.current_player = "O" if st.session_state.current_player == "X" else "X"
            elif st.session_state.game_mode == "1 Player (vs Computer)" and st.session_state.current_player == "X":
                st.session_state.current_player = "O"
                computer_index = computer_move(st.session_state.board)
                if computer_index is not None:
                    st.session_state.board[computer_index] = "O"
                    st.session_state.winner = check_winner(st.session_state.board)
                    st.session_state.current_player = "X"


def reset_game():
    st.session_state.board = [""] * 9
    st.session_state.current_player = "X"
    st.session_state.winner = None


st.subheader("Game Board")
cols = st.columns(3)
for i in range(9):
    with cols[i % 3]:
        button_label = st.session_state.board[i] if st.session_state.board[i] else " "
        st.button(button_label, key=f"btn_{i}", on_click=make_move, args=(i,))


if st.session_state.winner:
    if st.session_state.winner == "Draw":
        st.warning("It's a Draw!")
    else:
        st.success(f"Player {st.session_state.winner} Wins!")
else:
    st.info(f"Current Player: {st.session_state.current_player}")


st.button("Reset Game", on_click=reset_game, key="reset_btn")


st.markdown("""
<style>
    .stButton>button {
        width: 80px;
        height: 80px;
        font-size: 24px;
        border-radius: 10px;
        background-color: #f0f0f0;
        color: #333;
    }
    .stButton>button:hover {
        background-color: #ddd;
    }
    .stSelectbox {
        max-width: 200px;
        
    }
    div[data-baseweb="select"] > div {
        cursor: pointer;
    }
    .stButton[key="reset_btn"] {
        background-color: #ff4444;
        color: white;
        width: 120px;
    }
    .stButton[key="reset_btn"]:hover {
        background-color: #cc0000;
    }
</style>
""", unsafe_allow_html=True)