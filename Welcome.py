PAGE_TITLE = "Welcome"
import streamlit as st

st.set_page_config(page_title="Nintendo App", page_icon="🎮", layout="wide")

st.title("Welcome to the Nintendo Streamlit App!")

st.markdown(
    """
Select a section from the sidebar if you are a:
- **Gamer**: Receive game recommendations and explore game stats.
- **Developer**: Access detailed information about games that might make you a more successful developer.
"""
)
