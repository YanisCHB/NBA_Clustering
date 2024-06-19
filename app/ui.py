import streamlit as st

def set_page_container_style() -> None:
    """Set report container style."""
    margins_css = """
    <style>
        /* Configuration of paddings of containers inside main area */
        .main > div {
            max-width: 100%;
            padding-left: 5%;
        }

        /* Font size in tabs */
        button[data-baseweb="tab"] div p {
            font-size: 18px;
            font-weight: bold;
        }
    </style>
    """
    st.markdown(margins_css, unsafe_allow_html=True)

def sidebar():
    st.sidebar.title("NBA Player Stats Analysis")
    page = st.sidebar.radio("Navigation", ["Stats exploratoires", "Modélisation"])
    return page
