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

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #1D428A !important;
            color: white !important;
        }

        /* Sidebar text styling */
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3, 
        section[data-testid="stSidebar"] h4, 
        section[data-testid="stSidebar"] h5, 
        section[data-testid="stSidebar"] h6, 
        section[data-testid="stSidebar"] p, 
        section[data-testid="stSidebar"] div, 
        section[data-testid="stSidebar"] label, 
        section[data-testid="stSidebar"] span, 
        section[data-testid="stSidebar"] button {
            color: white !important;
        }

        /* Adjust font size for dropdown labels and tabs */
        .css-1a32fsj {  /* This targets dropdown labels */
            font-size: 16px !important;
        }

        .stTabs [role="tab"] {
            font-size: 18px !important;
            font-weight: bold !important;
        }

        /* Main page styling */
        .css-18e3th9 {
            background-color: white !important;
        }

        /* Button hover and active styles */
        section[data-testid="stSidebar"] button {
            background-color: #1D428A !important;
            border: none !important;
        }
        section[data-testid="stSidebar"] button:hover {
            background-color: #C8102E !important;
        }
        section[data-testid="stSidebar"] button:active {
            background-color: #C8102E !important;
        }
    </style>
    """
    st.markdown(margins_css, unsafe_allow_html=True)

def sidebar():
    st.sidebar.image("../nba.png", use_column_width=True)
    st.sidebar.title("NBA Player Stats Analysis")
    page = st.sidebar.radio("Navigation", ["Stats exploratoires", "Modélisation"])
    return page
